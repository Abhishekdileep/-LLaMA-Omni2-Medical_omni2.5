"""
Stage I(b) training: train the text-to-speech language model (MTTS).

MTTS is a decoder-only Transformer initialized from Qwen2.5-0.5B whose
vocabulary is extended with speech-unit tokens `<0> .. <unit_vocab_size-1>`
plus a `<sep>` token that marks the boundary between "reading" text and
"writing" the remaining speech units (mirrors `generate_units` in
`llama_omni2/model/speech_generator/speech_generator.py`, used at inference
time).

Per the paper, in Stage I(b) the gate fusion module is NOT trained: only
text embeddings (no LLM hidden states) are fed into MTTS. To emulate the
streaming "Read-R-Write-W" decoding strategy used at inference, we build a
single interleaved token sequence per example -- R text tokens, then W
speech-unit tokens, repeated until the text is exhausted, followed by a
`<sep>` token and the remaining speech units -- and compute the
cross-entropy loss only on the speech-unit (and `<sep>`/eos) positions.

Expected data format (`--data_path`), a JSON list of:
    {"text": "the text response", "units": [12, 4487, 233, ...]}
where `units` are the discrete speech tokens (0 <= u < unit_vocab_size)
obtained from the CosyVoice 2 speech tokenizer.
"""

import os
import json
import torch
import transformers
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from transformers import Qwen2ForCausalLM

from llama_omni2.constants import IGNORE_INDEX

SEP_TOKEN = "<sep>"


@dataclass
class ModelArguments:
    base_model_name_or_path: str = field(default="Qwen/Qwen2.5-0.5B")
    tts_tokenizer: str = field(
        default=None,
        metadata={"help": "Path to a tokenizer already extended with unit/<sep> tokens. "
                           "If unset, one is built from `base_model_name_or_path` and saved "
                           "to `<output_dir>/tts_tokenizer`."},
    )
    unit_vocab_size: int = field(default=6561)


@dataclass
class DataArguments:
    data_path: str = field(default="examples/tts_data.json")
    read_write_ratio: str = field(default="(3,10)", metadata={"help": "\"(R,W)\": read R text tokens, write W unit tokens."})


def unit_token(i: int) -> str:
    return f"<{i}>"


def build_tts_tokenizer(base_model_name_or_path: str, unit_vocab_size: int, save_path: str):
    """Extend the base Qwen2.5 tokenizer with speech-unit tokens and `<sep>`."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=False)
    new_tokens = [unit_token(i) for i in range(unit_vocab_size)] + [SEP_TOKEN]
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    return tokenizer


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

    # 1. Build (or load) the extended tokenizer for MTTS.
    tts_tokenizer_path = model_args.tts_tokenizer or os.path.join(training_args.output_dir, "tts_tokenizer")
    if model_args.tts_tokenizer and os.path.isdir(model_args.tts_tokenizer):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.tts_tokenizer, use_fast=False)
    else:
        tokenizer = build_tts_tokenizer(model_args.base_model_name_or_path, model_args.unit_vocab_size, tts_tokenizer_path)

    # 2. Initialize MTTS from Qwen2.5-0.5B and resize embeddings for the new vocab.
    model = Qwen2ForCausalLM.from_pretrained(model_args.base_model_name_or_path, torch_dtype=torch.bfloat16)
    for parameter in model.parameters():
        parameter.requires_grad = False
     
    model.resize_token_embeddings(len(tokenizer))
    for layer in model.model.layers[-1:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    model = model.cuda()



    # 3. Trainer.
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=TTSUnitDataset(data_path=data_args.data_path, tokenizer=tokenizer, read_write_ratio=data_args.read_write_ratio),
        data_collator=DataCollatorForTTSDataset(tokenizer=tokenizer),
    )

    # 4. Train.
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()

