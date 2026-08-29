"""Stage I(b) training: train the text-to-speech language model (MTTS).

MTTS is a decoder-only Transformer whose vocabulary is the LLM's, extended with
speech-unit tokens `<0> .. <unit_vocab_size-1>` plus a `<sep>` token that marks
the boundary between "reading" text and "writing" the remaining speech units
(mirrors `generate_units` in
`llama_omni2/model/speech_generator/speech_generator.py`, used at inference
time).

Per the paper, in Stage I(b) the gate fusion module is NOT trained: only text
embeddings (no LLM hidden states) are fed into MTTS. To emulate the streaming
"Read-R-Write-W" decoding strategy used at inference, we build a single
interleaved token sequence per example -- R text tokens, then W speech-unit
tokens, repeated until the text is exhausted, followed by a `<sep>` token and
the remaining speech units -- and compute the cross-entropy loss only on the
speech-unit (and `<sep>`/eos) positions.

Two things must line up with the rest of the stack, and neither is optional:

* **Token ids.** The unit ids MTTS emits are handed to CosyVoice 2, and the
  LLM's token ids are looked up directly in MTTS's embedding table
  (`generate_units` does exactly that). Both only work if this tokenizer is the
  checkpoint's `tts_tokenizer`, which is the 7B tokenizer with `<speech>`
  (151665) replaced by `<sep>` and `<0>..<6560>` appended at 151666..158226.
  Building a fresh tokenizer from `Qwen/Qwen2.5-0.5B` -- which has no
  `<speech>` -- puts `<0>` at 151665 and shifts **every** unit id by one.

* **Initialisation.** `--init_from_omni2` (default) starts MTTS from the
  checkpoint's `speech_generator.model.*`, which is already trained on unit
  prediction over this exact vocabulary. Starting from raw `Qwen/Qwen2.5-0.5B`
  leaves all 6561 unit embeddings random.

Expected data format (`--data_path`), a JSON list of:
    {"text": "the text response", "units": [12, 4487, 233, ...]}
where `units` are the discrete speech tokens (0 <= u < unit_vocab_size)
obtained from the CosyVoice 2 speech tokenizer.

    python llama_omni2/train_stage1b.py \
        --data_path examples/stage1b_data.json \
        --output_dir models/stage_1b \
        --bf16 True --num_train_epochs 10 --save_steps 500 --logging_steps 10
"""

import os
import json
import torch
import transformers
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from transformers import AutoConfig, Qwen2Config, Qwen2ForCausalLM

from llama_omni2.checkpoint_utils import read_prefixed_tensors
from llama_omni2.constants import IGNORE_INDEX
# Registers the omni2_speech2s_qwen2 config class, without which AutoConfig cannot
# read the checkpoint we take the tokenizer and the MTTS init from.
import llama_omni2.model  # noqa: F401

SEP_TOKEN = "<sep>"


@dataclass
class ModelArguments:
    omni2_path: str = field(
        default="models/LLaMA-Omni2-7B-Bilingual",
        metadata={"help": "2S checkpoint supplying the tts_tokenizer (and, with "
                          "--init_from_omni2, the MTTS weights)."},
    )
    init_from_omni2: bool = field(
        default=True,
        metadata={"help": "Warm-start MTTS from speech_generator.model.* instead of "
                          "raw Qwen2.5-0.5B. Its unit embeddings are already trained."},
    )
    base_model_name_or_path: str = field(
        default="Qwen/Qwen2.5-0.5B",
        metadata={"help": "Only used when --init_from_omni2 False."},
    )
    tts_tokenizer: str = field(
        default=None,
        metadata={"help": "Tokenizer already extended with unit/<sep> tokens. Defaults to "
                          "<omni2_path>/tts_tokenizer, which is the only layout the rest "
                          "of the stack agrees with."},
    )
    unit_vocab_size: int = field(default=6561)
    mtts_trainable_layers: int = field(
        default=-1,
        metadata={"help": "-1 trains all of MTTS (default; ~8 GB, comfortable on a 24 GB "
                          "card since the 7B is not loaded here). N > 0 trains only the "
                          "last N blocks plus the embedding."},
    )


@dataclass
class DataArguments:
    data_path: str = field(default="examples/tts_data.json")
    read_write_ratio: str = field(default=None, metadata={"help": "\"(R,W)\": read R text tokens, write W unit tokens. Defaults to the checkpoint's stream_params."})


def unit_token(i: int) -> str:
    return f"<{i}>"


def build_tts_tokenizer(base_model_name_or_path: str, unit_vocab_size: int, save_path: str):
    """Extend a base tokenizer with `<sep>` and the speech-unit tokens.

    Fallback only -- prefer the checkpoint's `tts_tokenizer`. `<sep>` is added
    FIRST because in the checkpoint it occupies 151665, the slot the 7B
    tokenizer gives `<speech>`; adding it last instead shifts every unit id by
    one and silently corrupts everything handed to CosyVoice 2.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=False)
    new_tokens = [SEP_TOKEN] + [unit_token(i) for i in range(unit_vocab_size)]
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    return tokenizer


def load_tts_tokenizer(model_args, output_dir: str):
    """Resolve the tokenizer, preferring the checkpoint's own."""
    explicit = model_args.tts_tokenizer
    if explicit and os.path.isdir(explicit):
        return transformers.AutoTokenizer.from_pretrained(explicit, use_fast=False)

    packaged = os.path.join(model_args.omni2_path, "tts_tokenizer")
    if os.path.isdir(packaged):
        return transformers.AutoTokenizer.from_pretrained(packaged, use_fast=False)

    print(f"WARNING: {packaged} not found; building a tokenizer from "
          f"{model_args.base_model_name_or_path}. Unit ids are only interchangeable with "
          "the pretrained checkpoint if it was built the same way.")
    return build_tts_tokenizer(model_args.base_model_name_or_path, model_args.unit_vocab_size,
                               os.path.join(output_dir, "tts_tokenizer"))


def build_mtts(model_args, tokenizer):
    """Instantiate MTTS, warm-started from the checkpoint's generator by default."""
    if model_args.init_from_omni2:
        config = AutoConfig.from_pretrained(model_args.omni2_path)
        model = Qwen2ForCausalLM(Qwen2Config(**config.speech_generator))
        state = read_prefixed_tensors(model_args.omni2_path, "speech_generator.model.")
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.endswith("lm_head.weight")]  # tied
        if missing or unexpected:
            raise RuntimeError(f"MTTS init mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        print(f"MTTS warm-started from {model_args.omni2_path} ({len(state)} tensors, "
              f"vocab {model.config.vocab_size})")
        if model.config.vocab_size != len(tokenizer):
            raise ValueError(
                f"vocab mismatch: checkpoint generator has {model.config.vocab_size}, "
                f"tokenizer has {len(tokenizer)}")
        return model

    model = Qwen2ForCausalLM.from_pretrained(model_args.base_model_name_or_path,
                                             torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    print(f"MTTS initialised from {model_args.base_model_name_or_path}; "
          f"{model_args.unit_vocab_size} unit embeddings are random")
    return model


class TTSUnitDataset(Dataset):
    """Builds interleaved (text, speech-unit) sequences for MTTS training."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, read_write_ratio: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.R, self.W = eval(read_write_ratio)
        with open(data_path, "r") as f:
            self.list_data_dict = json.load(f)
        self.sep_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.list_data_dict[i]
        text_ids = self.tokenizer.encode(item["text"], add_special_tokens=False)
        unit_ids = [self.tokenizer.convert_tokens_to_ids(unit_token(u)) for u in item["units"]]

        input_ids: List[int] = []
        is_unit: List[bool] = []

        ti, ui = 0, 0
        while ti < len(text_ids):
            text_chunk = text_ids[ti:ti + self.R]
            input_ids.extend(text_chunk)
            is_unit.extend([False] * len(text_chunk))
            ti += self.R

            unit_chunk = unit_ids[ui:ui + self.W]
            input_ids.extend(unit_chunk)
            is_unit.extend([True] * len(unit_chunk))
            ui += self.W

        # All text has been "read"; mark the boundary and flush the rest of the units.
        input_ids.append(self.sep_id)
        is_unit.append(True)
        remaining = unit_ids[ui:]
        input_ids.extend(remaining)
        is_unit.extend([True] * len(remaining))
        input_ids.append(self.eos_id)
        is_unit.append(True)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(
            [tok if flag else IGNORE_INDEX for tok, flag in zip(input_ids.tolist(), is_unit)],
            dtype=torch.long,
        )
        return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForTTSDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, transformers.TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 1. Resolve the tokenizer. This is the checkpoint's tts_tokenizer unless overridden,
    #    so unit ids stay interchangeable with CosyVoice 2 and with the 7B's token ids.
    tokenizer = load_tts_tokenizer(model_args, training_args.output_dir)

    # 2. Build MTTS.
    model = build_mtts(model_args, tokenizer)
    model = model.to(torch.bfloat16 if training_args.bf16 else torch.float32)

    # 3. Freeze per --mtts_trainable_layers. Note the unit embeddings must stay
    #    trainable whenever anything is: they are the only parameters that carry
    #    unit identity.
    if model_args.mtts_trainable_layers >= 0:
        model.requires_grad_(False)
        if model_args.mtts_trainable_layers > 0:
            model.get_input_embeddings().requires_grad_(True)
            for layer in model.model.layers[-model_args.mtts_trainable_layers:]:
                layer.requires_grad_(True)
    model = model.cuda()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M params")

    read_write_ratio = data_args.read_write_ratio
    if read_write_ratio is None:
        read_write_ratio = getattr(AutoConfig.from_pretrained(model_args.omni2_path),
                                   "stream_params", "(3,10)")
    print(f"read/write ratio {read_write_ratio}")

    # 4. Trainer.
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=TTSUnitDataset(data_path=data_args.data_path, tokenizer=tokenizer, read_write_ratio=read_write_ratio),
        data_collator=DataCollatorForTTSDataset(tokenizer=tokenizer),
    )

    # 5. Train.
    trainer.train()
    print(f"peak GPU memory {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
