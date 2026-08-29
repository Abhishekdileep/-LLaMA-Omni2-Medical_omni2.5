"""Stage I(a) training: speech adaptor + LLM, on <speech, text response> pairs.

Whisper is frozen; the speech adaptor and the last `--llm_trainable_layers`
blocks of the LLM are trained with cross-entropy against the gold text response.
On a 24 GB card only the last layer fits alongside the 7B in bf16, which is the
default.

Two things here are easy to get wrong and both were:

* **Loss masking.** Labels must be `IGNORE_INDEX` everywhere except the
  assistant turns, or the model is also trained to reproduce the system prompt
  and the user turn. The supervised span runs from the end of the generation
  prompt to and including `<|im_end|>` -- which is where generation actually
  stops -- excluding the template's trailing newline.

* **Saving.** Training happens on `Omni2SpeechQwen2ForCausalLM`, which is the
  class that computes a loss (`Omni2Speech2SQwen2ForCausalLM.forward` drops
  `labels` and always returns `loss=None`), and which does not allocate the
  0.5B speech generator.  But that class also *drops* `speech_generator.*` when
  it loads a 2S checkpoint, so saving it directly produces a checkpoint that
  cannot do speech-to-speech.  The final save therefore merges the trained
  tensors back over the source checkpoint, leaving the generator intact.

    python llama_omni2/train_stage1a.py \
        --model_name_or_path models/LLaMA-Omni2-7B-Bilingual \
        --data_path examples/questions.json \
        --output_dir models/stage1_output \
        --bf16 True --num_train_epochs 3 --per_device_train_batch_size 1 \
        --learning_rate 5e-5 --save_steps 500
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import whisper
import transformers
from safetensors.torch import save_file
from torch.utils.data import Dataset

from llama_omni2.checkpoint_utils import merge_and_save_checkpoint, prefixed_state_dict
from llama_omni2.model import Omni2SpeechQwen2ForCausalLM
from llama_omni2.constants import (
    DEFAULT_SPEECH_TOKEN, IGNORE_INDEX, MCQ_INSTRUCTION, SPEECH_TOKEN_INDEX)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="models/LLaMA-Omni2-7B-Bilingual")
    freeze_speech_encoder: bool = field(default=True)
    llm_trainable_layers: int = field(
        default=1,
        metadata={"help": "Train the last N LLM blocks. 1 is what fits on a 24 GB card "
                          "next to the 7B in bf16; 0 trains the speech adaptor alone."},
    )


@dataclass
class DataArguments:
    data_path: str = field(default="examples/questions.json")
    instruction: Optional[str] = field(
        default=None,
        metadata={"help": "Extra text appended to every user turn. Pass 'mcq' for the "
                          "shared MedQA prompt; anything else is used verbatim. Must "
                          "match what the model is served with at inference time."},
    )


class SupervisedSpeechDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer,
                 instruction: Optional[str] = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
        with open(data_path, "r") as f:
            self.list_data_dict = json.load(f)

    def load_speech(self, path):
        # Extract mel-spectrograms exactly as the inference code does
        speech = whisper.load_audio(path)
        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        return speech

    def __len__(self):
        return len(self.list_data_dict)

    def _assistant_spans(self, messages) -> List[tuple]:
        """Token spans to compute loss over, one per assistant turn.

        `apply_chat_template` is prefix-stable, so the span of turn k is
        [len(template(messages[:k], add_generation_prompt=True)),
         len(template(messages[:k+1])) - 1) -- the -1 drops the trailing newline
        the template emits after `<|im_end|>`.
        """
        spans = []
        for k, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            start = self.tokenizer.apply_chat_template(
                messages[:k], add_generation_prompt=True, return_tensors="pt").shape[1]
            end = self.tokenizer.apply_chat_template(
                messages[:k + 1], add_generation_prompt=False, return_tensors="pt").shape[1]
            spans.append((start, end - 1))
        return spans

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.list_data_dict[i]
        messages, speech_list = [], []

        # Build prompt & response
        for turn in item["conversation"]:
            if turn["from"] == "human":
                # `speech` may be a list: audio longer than whisper's 30 s window is
                # chunked, and each chunk gets its own <speech> token in the same turn.
                # prepare_inputs_labels_for_speech_and_text splices one 300-frame block
                # per token, in order.
                paths = turn["speech"]
                if isinstance(paths, str):
                    paths = [paths]
                content = DEFAULT_SPEECH_TOKEN * len(paths)
                if self.instruction:
                    content = content + "\n" + self.instruction
                messages.append({"role": "user", "content": content})
                speech_list.extend(self.load_speech(path) for path in paths)
            else:
                messages.append({"role": "assistant", "content": turn["text"]})

        # Tokenize (using chat template)
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, return_tensors="pt")[0]

        # Loss only on the assistant turns: without this the model is also trained to
        # reproduce the system prompt and the user turn.
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for start, end in self._assistant_spans(messages):
            labels[start:end] = input_ids[start:end]

        # Replace string token with the special SPEECH_TOKEN_INDEX
        input_ids[input_ids == self.speech_token_id] = SPEECH_TOKEN_INDEX

        return dict(input_ids=input_ids, labels=labels, speech=speech_list)


@dataclass
class DataCollatorForSpeechDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        all_speech = [speech for instance in instances for speech in instance["speech"]]

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        speech_tensors = torch.nn.utils.rnn.pad_sequence(all_speech, batch_first=True, padding_value=0)
        speech_lengths = torch.LongTensor([len(speech) for speech in all_speech])
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            speech=speech_tensors,
            speech_lengths=speech_lengths,
        )


class Stage1aTrainer(transformers.Trainer):
    """Saves a complete 2S checkpoint by merging over the source."""

    def __init__(self, *args, source_model_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_model_path = source_model_path
        self._full_save = True

    def save_model(self, output_dir=None, _internal_call=False):
        # A merged 2S checkpoint is ~16 GB; intermediate (--save_steps) checkpoints
        # keep only the trained tensors.
        self._full_save = not _internal_call
        super().save_model(output_dir=output_dir, _internal_call=_internal_call)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self._full_save:
            # `speech_generator.*` is absent from this model and so is copied through
            # from the source, keeping the output loadable by inference.py.
            merge_and_save_checkpoint(
                self.source_model_path, output_dir, prefixed_state_dict(self.model))
        else:
            trained = {n: p.detach().to("cpu").clone().contiguous()
                       for n, p in self.model.named_parameters() if p.requires_grad}
            save_file(trained, os.path.join(output_dir, "trainable.safetensors"))

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, transformers.TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 1. Load Model and Tokenizer
    #    Omni2SpeechQwen2ForCausalLM, not the 2S class: this one computes a loss and
    #    does not allocate the 0.5B speech generator. The generator weights are
    #    carried through at save time instead (see Stage1aTrainer._save).
    model = Omni2SpeechQwen2ForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=torch.bfloat16)
    model = model.cuda()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Freeze everything, then re-enable the speech adaptor and the last N LLM blocks.
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.get_speech_projector().requires_grad_(True)
    if model_args.llm_trainable_layers > 0:
        for layer in model.model.layers[-model_args.llm_trainable_layers:]:
            layer.requires_grad_(True)

    if not model_args.freeze_speech_encoder:
        model.get_speech_encoder().requires_grad_(True)
    print("Speech encoder is", "frozen." if model_args.freeze_speech_encoder else "trainable.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M params")

    # 3. Initialize Trainer
    trainer = Stage1aTrainer(
        model=model,
        args=training_args,
        train_dataset=SupervisedSpeechDataset(
            tokenizer=tokenizer, data_path=data_args.data_path,
            instruction=(MCQ_INSTRUCTION if data_args.instruction == "mcq"
                         else data_args.instruction)),
        data_collator=DataCollatorForSpeechDataset(tokenizer=tokenizer),
        source_model_path=model_args.model_name_or_path,
    )

    # 4. Train
    trainer.train()
    print(f"peak GPU memory {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
