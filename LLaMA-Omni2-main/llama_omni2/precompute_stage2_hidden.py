"""Cache the frozen LLM's hidden states for Stage II.

Stage II trains the gate fusion together with MTTS, which puts ~7 GB of weights,
gradients and optimizer moments on the GPU.  A co-resident 7B (15.2 GB) plus
whisper (1.3 GB) does not leave room for that on a 24 GB card, so the upstream
is run once here and its output written to disk.  The upstream is frozen, so
`llm_h` is a deterministic function of (audio, text) and caching it is exactly
equivalent to recomputing it -- with the one caveat that **the cache is
invalidated by any Stage I(a) fine-tune** and must be rebuilt against the new
LLM before Stage II runs again.

Alignment.  `generation.py::_sample` pairs each hidden state with the token it
produced: on the first TTS call it builds [h_{P-1}, h_P, h_{P+1}] and pairs them
with the first three generated tokens.  So with prompt length P (after the 300
speech frames have been spliced in) and gold response y_1..y_L at positions
P..P+L-1, the slice this script stores is

    llm_h = H[P-1 : P+L-1]      against      resp_ids = input_ids[P : P+L]

Both length L. Getting this off by one trains Stage II on misaligned pairs
without any error, which is why `train_stage2.py` reads the slice rather than
recomputing it.

Run from the LLaMA-Omni2-main directory (config.speech_encoder is a relative path).

    python llama_omni2/precompute_stage2_hidden.py \
        --model_name_or_path models/LLaMA-Omni2-7B-Bilingual \
        --data_path examples/stage2_data.json \
        --cache_dir cache/stage2_hidden
"""

import argparse
import json
import os

import numpy as np
import torch
import transformers
import whisper
from transformers import AutoConfig

from llama_omni2.constants import (
    DEFAULT_SPEECH_TOKEN, MCQ_INSTRUCTION, SPEECH_TOKEN_INDEX)
from llama_omni2.model import Omni2SpeechQwen2ForCausalLM


def load_speech(path):
    """Log-mel features, extracted exactly as train_stage1a.py and inference.py do."""
    speech = whisper.load_audio(path)
    speech = whisper.pad_or_trim(speech)
    return whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)


def build_ids(tokenizer, text, instruction=None, n_speech=1):
    """Return (prompt_ids, resp_ids) for one single-turn example.

    The prompt is built with `add_generation_prompt=True` so it is byte-identical
    to what `OmniModel.process_messages` produces at inference; the response is
    the gold text plus `<|im_end|>`, which is where generation actually stops.
    `n_speech` > 1 for audio chunked across whisper's 30 s window.
    """
    content = DEFAULT_SPEECH_TOKEN * n_speech
    if instruction:
        content = content + "\n" + instruction
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, return_tensors="pt")[0]
    prompt_ids[prompt_ids == tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)] = SPEECH_TOKEN_INDEX

    resp_ids = tokenizer.encode(text, add_special_tokens=False)
    resp_ids.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    return prompt_ids, torch.tensor(resp_ids, dtype=torch.long)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="models/LLaMA-Omni2-7B-Bilingual")
    parser.add_argument("--data_path", default="examples/stage2_data.json")
    parser.add_argument("--cache_dir", default="cache/stage2_hidden")
    parser.add_argument("--instruction", choices=["none", "mcq"], default="none",
                        help="Extra user-turn instruction. Use 'mcq' for MedQA, so the "
                             "cached prompt matches inference.py's MCQ_INSTRUCTION.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    with open(args.data_path) as f:
        records = json.load(f)

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    model = Omni2SpeechQwen2ForCausalLM.from_pretrained(
        args.model_name_or_path, config=config, torch_dtype=torch.bfloat16)
    model = model.cuda().eval()

    instruction = MCQ_INSTRUCTION if args.instruction == "mcq" else None
    written, skipped = 0, 0

    for record in records:
        out_path = os.path.join(args.cache_dir, f"{record['id']}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        paths = record["speech"]
        if isinstance(paths, str):
            paths = [paths]

        prompt_ids, resp_ids = build_ids(tokenizer, record["text"], instruction, len(paths))
        input_ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0).cuda()

        speech = torch.stack([load_speech(path) for path in paths]).to(
            dtype=torch.bfloat16, device="cuda")
        speech_lengths = torch.LongTensor([speech.shape[1]] * len(paths)).cuda()

        # Splices the 300 projected speech frames in at the SPEECH_TOKEN_INDEX position,
        # so the prompt length is only known after this call.
        (_, position_ids, attention_mask, _, inputs_embeds, _
         ) = model.prepare_inputs_labels_for_speech_and_text(
            input_ids, None, None, None, None, speech, speech_lengths)

        # `hidden_states[-1]` of the causal LM is `last_hidden_state` of the base model
        # (Qwen2Model appends the post-norm states last), so this skips the 152k-way
        # lm_head projection that Stage II never uses.
        hidden = model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                             position_ids=position_ids).last_hidden_state

        n_text = resp_ids.shape[0]
        prompt_len = inputs_embeds.shape[1] - n_text
        llm_h = hidden[0, prompt_len - 1: prompt_len + n_text - 1]

        assert llm_h.shape[0] == n_text, f"{record['id']}: {llm_h.shape[0]} != {n_text}"
        np.savez(out_path,
                 llm_h=llm_h.to(torch.float16).cpu().numpy(),
                 resp_ids=resp_ids.numpy().astype(np.int32))
        written += 1
        if written % 25 == 0:
            print(f"  {written} written (prompt_len={prompt_len}, L={n_text})")

    print(f"wrote {written}, skipped {skipped} -> {args.cache_dir}")


if __name__ == "__main__":
    main()
