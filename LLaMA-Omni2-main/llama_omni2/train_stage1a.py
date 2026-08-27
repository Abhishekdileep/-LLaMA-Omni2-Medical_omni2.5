import os
import json
import torch
import whisper
import transformers
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from llama_omni2.model import Omni2SpeechQwen2ForCausalLM
from llama_omni2.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, IGNORE_INDEX

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="models/LLaMA-Omni2-7B-Bilingual")
    freeze_speech_encoder: bool = field(default=True)

@dataclass
class DataArguments:
    data_path: str = field(default="examples/questions.json")

class SupervisedSpeechDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
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

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.list_data_dict[i]
        messages, speech_list = [], []
        
        # Build prompt & response
        for turn in item["conversation"]:
            if turn["from"] == "human":
                messages.append({"role": "user", "content": DEFAULT_SPEECH_TOKEN})
                speech_list.append(self.load_speech(turn["speech"]))
            else:
                messages.append({"role": "assistant", "content": turn["text"]})
        
        # Tokenize (using chat template)
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt")[0]
        
        # Replace string token with the special SPEECH_TOKEN_INDEX
        input_ids[input_ids == self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)] = SPEECH_TOKEN_INDEX
        
        # In a production script, you must mask out the user's prompt (setting those tokens to IGNORE_INDEX)
        # so you only compute loss on the assistant's text!
        labels = input_ids.clone() 
        
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

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, transformers.TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 1. Load Model and Tokenizer
    model = Omni2SpeechQwen2ForCausalLM.from_pretrained(model_args.model_name_or_path ,     torch_dtype=torch.bfloat16)
    model = model.cuda()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )

    # 2. Freeze the Speech Encoder (Whisper) for Stage 1a
    if model_args.freeze_speech_encoder:
        model.get_speech_encoder().requires_grad_(False)
        print("Speech encoder is frozen.")
    
    # 3. Ensure Speech Adaptor and LLM are trainable
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.get_speech_projector().requires_grad_(True)

    for layer in model.model.layers[-1:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    model.get_speech_projector().requires_grad_(True)
    # model.model.requires_grad_(True)

    # 4. Initialize Trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=SupervisedSpeechDataset(tokenizer=tokenizer, data_path=data_args.data_path),
        data_collator=DataCollatorForSpeechDataset(tokenizer=tokenizer),
    )

    # 5. Train
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()1~import os
import json
import torch
import whisper
import transformers
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from llama_omni2.model import Omni2SpeechQwen2ForCausalLM
from llama_omni2.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, IGNORE_INDEX

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="models/LLaMA-Omni2-7B-Bilingual")
    freeze_speech_encoder: bool = field(default=True)

@dataclass
class DataArguments:
    data_path: str = field(default="examples/questions.json")

class SupervisedSpeechDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
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

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.list_data_dict[i]
        messages, speech_list = [], []
        
        # Build prompt & response
        for turn in item["conversation"]:
            if turn["from"] == "human":
                messages.append({"role": "user", "content": DEFAULT_SPEECH_TOKEN})
                speech_list.append(self.load_speech(turn["speech"]))
            else:
                messages.append({"role": "assistant", "content": turn["text"]})
        
        # Tokenize (using chat template)
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, return_tensors="pt")[0]
        
        # Replace string token with the special SPEECH_TOKEN_INDEX
        input_ids[input_ids == self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)] = SPEECH_TOKEN_INDEX
        
        # In a production script, you must mask out the user's prompt (setting those tokens to IGNORE_INDEX)
        # so you only compute loss on the assistant's text!
        labels = input_ids.clone() 
        
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

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, transformers.TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 1. Load Model and Tokenizer
    model = Omni2SpeechQwen2ForCausalLM.from_pretrained(model_args.model_name_or_path ,     torch_dtype=torch.bfloat16)
    model = model.cuda()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )

    # 2. Freeze the Speech Encoder (Whisper) for Stage 1a
    if model_args.freeze_speech_encoder:
        model.get_speech_encoder().requires_grad_(False)
        print("Speech encoder is frozen.")
    
    # 3. Ensure Speech Adaptor and LLM are trainable
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.get_speech_projector().requires_grad_(True)

    for layer in model.model.layers[-1:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    model.get_speech_projector().requires_grad_(True)
    # model.model.requires_grad_(True)

    # 4. Initialize Trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=SupervisedSpeechDataset(tokenizer=tokenizer, data_path=data_args.data_path),
        data_collator=DataCollatorForSpeechDataset(tokenizer=tokenizer),
    )

    # 5. Train
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
