"""Checks for the Stage II stack. Run from the LLaMA-Omni2-main directory.

    python llama_omni2/verify_stage2.py --model_name_or_path models/LLaMA-Omni2-7B-Bilingual

1. fusion warm-start   -- GateFusionModule is bit-exact with, and computes the
                          same function as, the checkpoint's input_proj + gate.
2. tokenizer alignment -- <0>=151666, <sep>=151665, and the shared prefix ids
                          agree with the 7B text tokenizer (this is what makes
                          MTTS embedding lookups on LLM token ids legal).
3. hidden alignment    -- the cached llm_h slice matches what the real
                          `generate` path feeds to `generate_units`, and a +-1
                          shift does not. This is the check that catches the
                          off-by-one; cross-entropy cannot, because the shipped
                          gate is nearly closed (see check 4).
4. gate statistics     -- reports how much of the fused representation actually
                          comes from the LLM hidden states.
"""

import argparse
import os

import numpy as np
import torch
import transformers
import whisper
from transformers import AutoConfig

from llama_omni2.constants import DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX
from llama_omni2.fusion import GateFusionModule
from llama_omni2.train_stage2 import load_speech_generator

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  --  {detail}" if detail else ""))


def check_fusion(model_path):
    print("\n1. fusion warm-start")
    config, gen = load_speech_generator(model_path, dtype=torch.float32)
    fusion = GateFusionModule.from_pretrained_generator(gen, dtype=torch.float32)

    pairs = [(gen.input_proj[0], fusion.ffn[0]),
             (gen.input_proj[2], fusion.ffn[2]),
             (gen.gate[0], fusion.gate_linear)]
    exact = all(torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias) for a, b in pairs)
    check("6/6 tensors bit-exact vs pretrained input_proj/gate", exact)

    h = torch.randn(2, 7, config.hidden_size)
    e = torch.randn(2, 7, gen.model.config.hidden_size)
    delta = (fusion(h, e) - gen.fusion(gen.input_proj(h), e)).abs().max().item()
    check("same function as speech_generator.fusion", delta < 1e-5, f"max|diff| = {delta:.2e}")

    # Round-trip: write back and confirm the checkpoint layout is restored exactly.
    with torch.no_grad():
        fusion.ffn[0].weight.add_(1.0)
    fusion.write_back_to_generator(gen)
    check("write_back_to_generator round-trips",
          torch.equal(gen.input_proj[0].weight, fusion.ffn[0].weight))
    return config


def check_tokenizers(model_path, config):
    print("\n2. tokenizer alignment")
    tts = transformers.AutoTokenizer.from_pretrained(config.tts_tokenizer, use_fast=False)
    txt = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False)
    check("<0> == 151666", tts.convert_tokens_to_ids("<0>") == 151666,
          f"got {tts.convert_tokens_to_ids('<0>')}")
    check("<sep> == 151665", tts.convert_tokens_to_ids("<sep>") == 151665,
          f"got {tts.convert_tokens_to_ids('<sep>')}")
    check("unit ids span the whole vocabulary",
          tts.convert_tokens_to_ids(f"<{config.unit_vocab_size - 1}>") is not None)
    probe = "The patient presents with acute pain, 42 mg/dL."
    check("shared prefix ids identical to the 7B tokenizer",
          txt.encode(probe, add_special_tokens=False) == tts.encode(probe, add_special_tokens=False))


def check_alignment(model_path, wav, cache_npz):
    print("\n3. hidden-state alignment vs the live inference path")
    if not os.path.isfile(cache_npz):
        check("cache present", False, f"{cache_npz} missing -- run precompute_stage2_hidden.py")
        return
    from llama_omni2.model import Omni2Speech2SQwen2ForCausalLM

    cfg = AutoConfig.from_pretrained(model_path)
    cfg.tts_tokenizer = os.path.join(model_path, "tts_tokenizer")
    tok = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = Omni2Speech2SQwen2ForCausalLM.from_pretrained(
        model_path, config=cfg, torch_dtype=torch.bfloat16).cuda().eval()

    captured = []
    original = model.speech_generator.generate_units

    def spy(tts_inputs, new_hidden_states, new_tokens, is_finished=False):
        captured.append((new_hidden_states.detach().float().cpu(), new_tokens.detach().cpu()))
        return original(tts_inputs, new_hidden_states, new_tokens, is_finished)

    model.speech_generator.generate_units = spy

    ids = tok.apply_chat_template([{"role": "user", "content": DEFAULT_SPEECH_TOKEN}],
                                  add_generation_prompt=True, return_tensors="pt")[0]
    ids[ids == tok.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)] = SPEECH_TOKEN_INDEX
    mel = whisper.log_mel_spectrogram(
        whisper.pad_or_trim(whisper.load_audio(wav)), n_mels=128).permute(1, 0)
    model.generate(inputs=ids.unsqueeze(0).cuda(),
                   speech=mel.unsqueeze(0).to(torch.bfloat16).cuda(),
                   speech_lengths=torch.LongTensor([mel.shape[0]]).cuda(),
                   do_sample=False, max_new_tokens=512, use_cache=True)

    live_h = torch.cat([h for h, _ in captured], 0)
    live_t = torch.cat([t for _, t in captured], 0)
    cached = np.load(cache_npz)
    cache_h = torch.from_numpy(cached["llm_h"]).float()
    cache_t = torch.from_numpy(cached["resp_ids"]).long()

    n = min(len(live_t), len(cache_t))
    check("greedy decode reproduces the cached response tokens",
          torch.equal(live_t[:n], cache_t[:n]), f"{n} tokens")
    rel = ((live_h[:n] - cache_h[:n]).norm() / cache_h[:n].norm()).item()
    check("cached hidden states match the live pairs", rel < 0.05,
          f"relative error {rel:.5f} (bf16 noise floor ~1e-2)")
    shifted = min(((live_h[:n - 1] - torch.roll(cache_h[:n], s, 0)[:n - 1]).norm()
                   / cache_h[:n - 1].norm()).item() for s in (1, -1))
    check("a +-1 shift is clearly worse", shifted > 10 * rel,
          f"shifted error {shifted:.5f} vs aligned {rel:.5f}")

    del model
    torch.cuda.empty_cache()


def check_gate(model_path, cache_npz):
    print("\n4. gate statistics (diagnostic, not pass/fail)")
    if not os.path.isfile(cache_npz):
        print("     skipped, no cache")
        return
    _, gen = load_speech_generator(model_path, dtype=torch.float32)
    fusion = GateFusionModule.from_pretrained_generator(gen, dtype=torch.float32)
    cached = np.load(cache_npz)
    h = torch.from_numpy(cached["llm_h"]).float().unsqueeze(0)
    e = gen.model.get_input_embeddings()(torch.from_numpy(cached["resp_ids"]).long().unsqueeze(0)).float()
    g = torch.sigmoid(fusion.gate_linear(torch.cat([fusion.ffn(h), e], -1)))
    share = ((fusion(h, e) - e).norm() / e.norm()).item()
    print(f"     gate mean {g.mean():.4f}, fraction below 0.01: {(g < 0.01).float().mean():.3f}")
    print(f"     ||fused - text_emb|| / ||text_emb|| = {share:.4f}")
    print("     A near-zero gate means the shipped checkpoint barely uses the LLM hidden")
    print("     states, so Stage II has real headroom -- and cross-entropy alone cannot")
    print("     detect a misalignment, which is why check 3 compares tensors directly.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="models/LLaMA-Omni2-7B-Bilingual")
    parser.add_argument("--wav", default="examples/wav/helpful_base_0.wav")
    parser.add_argument("--cache_npz", default="cache/stage2_hidden/helpful_base_0.npz")
    parser.add_argument("--skip_alignment", action="store_true",
                        help="Skip check 3, which loads the full 7B.")
    args = parser.parse_args()

    config = check_fusion(args.model_name_or_path)
    check_tokenizers(args.model_name_or_path, config)
    if not args.skip_alignment:
        check_alignment(args.model_name_or_path, args.wav, args.cache_npz)
    check_gate(args.model_name_or_path, args.cache_npz)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
