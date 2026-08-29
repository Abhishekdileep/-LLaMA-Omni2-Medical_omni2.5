"""Stage II training: gate fusion + MTTS, against a frozen speech-to-text stack.

Stage I(b) trains MTTS on text embeddings alone.  Stage II swaps those text
embeddings for the *fused* representation of Eq. 1-4 -- the LLM's hidden state
projected into the TTS space and gated against the text embedding -- so MTTS
gets the semantic content the text tokens alone do not carry.  The gate fusion
and MTTS train together: the fusion changes what MTTS is fed, so MTTS has to
move with it.

    frozen : whisper encoder, speech projector, LLM (all 28 layers)
    trained: GateFusionModule (33.7M) + the MTTS decoder (~500M)

Because the upstream is frozen, its hidden states are deterministic, so they are
computed once by `precompute_stage2_hidden.py` and read from a cache here.  That
is not just a speed-up: with MTTS trainable, a co-resident 7B does not fit in
24 GB (see the memory table in the README).  Nothing in this script loads the
LLM or whisper -- only `speech_generator.*` is pulled out of the checkpoint.

Alignment invariant (mirrors `generation.py::_sample`, which pairs the hidden
states with the tokens they produced):

    h_i is the state that PRODUCED y_i

so with prompt length P and gold response y_1..y_L the training slices are
`H[:, P-1 : P+L-1]` against `input_ids[:, P : P+L]`.  The cache stores that
slice already aligned; see `precompute_stage2_hidden.py`.

The MTTS input sequence is the same interleaved Read-R-Write-W layout as Stage
I(b) (`train_stage1b.py`), with the R text slots carrying fused embeddings
instead of raw text embeddings, and cross-entropy computed only on the
speech-unit (and `<sep>`/eos) positions.

Expected data format (`--data_path`), a JSON list of:
    {"id": "...", "speech": "path/to/question.wav",
     "text": "the text response", "units": [12, 4487, 233, ...]}
"""

import os
import json
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import transformers
from safetensors.torch import save_file
from torch.utils.data import Dataset
from transformers import AutoConfig
from transformers.trainer_pt_utils import get_parameter_names

from llama_omni2.checkpoint_utils import (
    merge_and_save_checkpoint, prefixed_state_dict, read_prefixed_tensors)
from llama_omni2.constants import IGNORE_INDEX
from llama_omni2.fusion import GateFusionModule
from llama_omni2.model.speech_generator.speech_generator import LLMSpeechGenerator

SEP_TOKEN = "<sep>"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="models/LLaMA-Omni2-7B-Bilingual",
        metadata={"help": "2S checkpoint supplying the config, the tts_tokenizer and the "
                          "pretrained speech_generator. Also the source the final merged "
                          "checkpoint is written from, so point this at the Stage I(a) "
                          "output when fine-tuning."},
    )
    mtts_init: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Stage I(b) output to initialise MTTS from, so Stage II "
                          "continues that run instead of resetting to the pretrained "
                          "generator. The gate fusion still warm-starts from the checkpoint."},
    )
    mtts_trainable_layers: int = field(
        default=-1,
        metadata={"help": "-1 trains all of MTTS (default). N > 0 trains only its last N "
                          "blocks plus the embedding; 0 freezes MTTS and trains the fusion "
                          "alone. Memory escape hatch."},
    )


@dataclass
class DataArguments:
    data_path: str = field(default="examples/stage2_data.json")
    hidden_cache_dir: str = field(
        default="cache/stage2_hidden",
        metadata={"help": "Directory written by precompute_stage2_hidden.py."},
    )


@dataclass
class Stage2TrainingArguments(transformers.TrainingArguments):
    fusion_lr: float = field(
        default=1e-4,
        metadata={"help": "Learning rate for the gate fusion. Kept separate from "
                          "--learning_rate because a 33.7M adapter and a 500M pretrained "
                          "decoder do not want the same step size."},
    )


# ---------------------------------------------------------------- checkpoint I/O


def load_speech_generator(model_path: str, mtts_init: Optional[str] = None,
                          dtype: torch.dtype = torch.bfloat16):
    """Build an `LLMSpeechGenerator` and load only `speech_generator.*` into it.

    Returns (config, generator). The LLM and whisper tensors in the checkpoint
    are never touched -- that is what keeps Stage II inside 24 GB.
    """
    config = AutoConfig.from_pretrained(model_path)
    config.tts_tokenizer = os.path.join(model_path, "tts_tokenizer")

    generator = LLMSpeechGenerator(config)
    state = read_prefixed_tensors(model_path, "speech_generator.")
    if not state:
        raise RuntimeError(
            f"{model_path} has no speech_generator.* weights. If this is a Stage I(a) "
            "output, it was written by the pre-fix train_stage1a.py which dropped them; "
            "re-run Stage I(a) with the current script."
        )
    missing, unexpected = generator.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected speech_generator keys: {unexpected[:5]}")
    # `model.lm_head.weight` is tied to the embedding, so it is legitimately absent.
    missing = [k for k in missing if not k.endswith("lm_head.weight")]
    if missing:
        raise RuntimeError(f"Missing speech_generator keys: {missing[:5]}")

    if mtts_init:
        mtts_state = read_prefixed_tensors(mtts_init, "", strip=False)
        generator.model.load_state_dict(mtts_state, strict=False)
        print(f"MTTS initialised from {mtts_init} ({len(mtts_state)} tensors)")

    return config, generator.to(dtype)


def save_generator_only(output_dir: str, generator) -> None:
    """Light intermediate checkpoint: the trained part and nothing else."""
    os.makedirs(output_dir, exist_ok=True)
    save_file(prefixed_state_dict(generator, "speech_generator."),
              os.path.join(output_dir, "speech_generator.safetensors"))


def save_2s_checkpoint(source_model_path: str, output_dir: str, generator) -> None:
    """Write a complete 2S checkpoint: source shards with speech_generator.* replaced."""
    merge_and_save_checkpoint(source_model_path, output_dir,
                              prefixed_state_dict(generator, "speech_generator."))


# ---------------------------------------------------------------- data


class Stage2Dataset(Dataset):
    """Interleaved (fused-text, speech-unit) sequences for gate-fusion training.

    Emits *slots* rather than embeddings: each position is either a text slot
    (an index into this example's cached hidden states) or a unit slot (a token
    id). The fused embeddings are built in the forward pass, where the fusion
    module's gradients live.
    """

    def __init__(self, data_path: str, hidden_cache_dir: str,
                 tokenizer: transformers.PreTrainedTokenizer, read_write_ratio: str):
        super().__init__()
        self.cache_dir = hidden_cache_dir
        self.R, self.W = eval(read_write_ratio)
        with open(data_path, "r") as f:
            self.list_data_dict = json.load(f)

        self.sep_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        self.eos_id = tokenizer.eos_token_id
        self.unit_base = tokenizer.convert_tokens_to_ids("<0>")
        if self.unit_base is None or self.sep_id is None:
            raise ValueError("tts_tokenizer is missing <0> / <sep>; see build_tts_tokenizer.")

        missing = [r for r in self.list_data_dict if not os.path.isfile(self._cache_path(r))]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(self.list_data_dict)} examples have no cached hidden "
                f"states (first: {self._cache_path(missing[0])}). Run "
                "precompute_stage2_hidden.py against this --data_path first."
            )

    def _cache_path(self, record) -> str:
        return os.path.join(self.cache_dir, f"{record['id']}.npz")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        record = self.list_data_dict[i]
        cached = np.load(self._cache_path(record))
        llm_h = torch.from_numpy(cached["llm_h"])                      # (L, llm_hidden)
        resp_ids = torch.from_numpy(cached["resp_ids"]).long()         # (L,)
        n_text = resp_ids.shape[0]

        # Speech-unit ids live at <0>.. in the TTS vocabulary.
        unit_ids = [self.unit_base + int(u) for u in record["units"]]

        is_unit: List[bool] = []
        text_pos: List[int] = []   # index into llm_h/resp_ids; 0 on unit slots
        unit_tok: List[int] = []   # token id; 0 on text slots
        labels: List[int] = []

        def push_unit(token_id: int):
            is_unit.append(True)
            text_pos.append(0)
            unit_tok.append(token_id)
            labels.append(token_id)

        ti, ui = 0, 0
        while ti < n_text:
            for j in range(ti, min(ti + self.R, n_text)):
                is_unit.append(False)
                text_pos.append(j)
                unit_tok.append(0)
                labels.append(IGNORE_INDEX)   # loss only on unit positions
            ti += self.R

            for token_id in unit_ids[ui:ui + self.W]:
                push_unit(token_id)
            ui += self.W

        # All text has been "read"; mark the boundary and flush the rest of the units.
        # `<sep>` is force-fed at inference (speech_generator.generate_units), but it is
        # still supervised here -- same as Stage I(b) -- as a free boundary signal.
        push_unit(self.sep_id)
        for token_id in unit_ids[ui:]:
            push_unit(token_id)
        push_unit(self.eos_id)

        return dict(
            llm_h=llm_h,
            resp_ids=resp_ids,
            is_unit=torch.tensor(is_unit, dtype=torch.bool),
            text_pos=torch.tensor(text_pos, dtype=torch.long),
            unit_tok=torch.tensor(unit_tok, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
        )


@dataclass
class DataCollatorForStage2Dataset:
    """Right-pads both the per-example text axis (L) and the slot axis (S).

    Right padding matches `tokenizer_padding_side: right` in the checkpoint config
    and the Stage I collators; padded slots are masked out of both the attention
    mask and the labels, so the side is immaterial to the loss.
    """

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        llm_h = torch.nn.utils.rnn.pad_sequence(
            [x["llm_h"] for x in instances], batch_first=True, padding_value=0.0)
        resp_ids = torch.nn.utils.rnn.pad_sequence(
            [x["resp_ids"] for x in instances], batch_first=True, padding_value=0)
        is_unit = torch.nn.utils.rnn.pad_sequence(
            [x["is_unit"] for x in instances], batch_first=True, padding_value=False)
        text_pos = torch.nn.utils.rnn.pad_sequence(
            [x["text_pos"] for x in instances], batch_first=True, padding_value=0)
        unit_tok = torch.nn.utils.rnn.pad_sequence(
            [x["unit_tok"] for x in instances], batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(
            [x["labels"] for x in instances], batch_first=True, padding_value=IGNORE_INDEX)

        lengths = torch.tensor([x["labels"].shape[0] for x in instances])
        attention_mask = torch.arange(labels.shape[1])[None, :] < lengths[:, None]

        return dict(llm_h=llm_h, resp_ids=resp_ids, is_unit=is_unit, text_pos=text_pos,
                    unit_tok=unit_tok, labels=labels, attention_mask=attention_mask)


# ---------------------------------------------------------------- model


class Stage2Model(nn.Module):
    """Gate fusion + MTTS. The frozen upstream is represented only by `llm_h`."""

    def __init__(self, generator, fusion: GateFusionModule):
        super().__init__()
        self.generator = generator
        self.fusion = fusion
        # `input_proj` / `gate` hold the same function as `fusion` and would otherwise
        # be duplicated in the optimizer; they are refreshed from `fusion` on save.
        self.generator.input_proj.requires_grad_(False)
        self.generator.gate.requires_grad_(False)
        self.config = generator.model.config

    @property
    def mtts(self):
        return self.generator.model

    def gradient_checkpointing_enable(self, **kwargs):
        self.mtts.gradient_checkpointing_enable(**kwargs)

    def forward(self, llm_h, resp_ids, is_unit, text_pos, unit_tok, labels, attention_mask):
        embed = self.mtts.get_input_embeddings()
        text_emb = embed(resp_ids)                                        # (B, L, d)

        # Eq 1-4. Run in fp32: the gate is a sigmoid over a difference of two
        # bf16 projections, which is exactly where bf16 loses the most.
        fused = self.fusion(llm_h.float(), text_emb.float()).to(text_emb.dtype)

        d = fused.shape[-1]
        text_slots = torch.gather(fused, 1, text_pos.unsqueeze(-1).expand(-1, -1, d))
        unit_slots = embed(unit_tok)
        inputs_embeds = torch.where(is_unit.unsqueeze(-1), unit_slots, text_slots)

        return self.mtts(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                         labels=labels)


class Stage2Trainer(transformers.Trainer):
    """Two learning rates, and a save path that reassembles a full 2S checkpoint."""

    def __init__(self, *args, source_model_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_model_path = source_model_path
        self._full_save = True

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        decay = set(get_parameter_names(self.model, [nn.LayerNorm]))
        decay = {n for n in decay if "bias" not in n}

        def collect(is_fusion, is_decay):
            return [p for n, p in self.model.named_parameters()
                    if p.requires_grad
                    and n.startswith("fusion.") == is_fusion
                    and (n in decay) == is_decay]

        cls_, kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(self.args)
        kwargs.pop("lr", None)
        groups = [
            {"params": collect(False, True), "lr": self.args.learning_rate,
             "weight_decay": self.args.weight_decay},
            {"params": collect(False, False), "lr": self.args.learning_rate,
             "weight_decay": 0.0},
            {"params": collect(True, True), "lr": self.args.fusion_lr,
             "weight_decay": self.args.weight_decay},
            {"params": collect(True, False), "lr": self.args.fusion_lr,
             "weight_decay": 0.0},
        ]
        self.optimizer = cls_([g for g in groups if g["params"]], **kwargs)
        return self.optimizer

    def save_model(self, output_dir=None, _internal_call=False):
        # Intermediate (--save_steps) checkpoints keep only the trained part; a full
        # merged 2S checkpoint is ~16 GB and is only worth writing at the end.
        self._full_save = not _internal_call
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model = self.model
        model.fusion.write_back_to_generator(model.generator)

        if self._full_save:
            save_2s_checkpoint(self.source_model_path, output_dir, model.generator)
        else:
            save_generator_only(output_dir, model.generator)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, Stage2TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    dtype = torch.bfloat16 if training_args.bf16 else torch.float32

    # 1. Pull only the speech generator out of the 2S checkpoint. The LLM and whisper
    #    weights stay on disk; their contribution arrives through the hidden-state cache.
    config, generator = load_speech_generator(
        model_args.model_name_or_path, mtts_init=model_args.mtts_init, dtype=dtype)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.tts_tokenizer, use_fast=False)

    # 2. Warm-start the gate fusion from the pretrained input_proj / gate.
    fusion = GateFusionModule.from_pretrained_generator(generator, dtype=torch.float32)

    model = Stage2Model(generator, fusion).cuda()

    # 3. Freeze per --mtts_trainable_layers (-1 = train all of MTTS).
    if model_args.mtts_trainable_layers >= 0:
        model.mtts.requires_grad_(False)
        if model_args.mtts_trainable_layers > 0:
            model.mtts.get_input_embeddings().requires_grad_(True)
            for layer in model.mtts.model.layers[-model_args.mtts_trainable_layers:]:
                layer.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M params")

    # 4. Trainer.
    trainer = Stage2Trainer(
        model=model,
        args=training_args,
        train_dataset=Stage2Dataset(
            data_path=data_args.data_path,
            hidden_cache_dir=data_args.hidden_cache_dir,
            tokenizer=tokenizer,
            read_write_ratio=config.stream_params,
        ),
        data_collator=DataCollatorForStage2Dataset(),
        source_model_path=model_args.model_name_or_path,
    )

    # 5. Train.
    trainer.train()
    print(f"peak GPU memory {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
