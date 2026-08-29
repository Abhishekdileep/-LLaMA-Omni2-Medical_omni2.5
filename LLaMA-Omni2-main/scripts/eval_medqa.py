"""Run a model over a MedQA manifest and write predictions for bleuscore.py.

`run.sh` launches inference.py once per wav, which reloads the 17 GB checkpoint
every time. This loads it once and walks the whole split, and -- unlike
inference.py's `main` -- resets the history between items, so each question is
answered independently instead of being conditioned on the previous answers.

    python scripts/eval_medqa.py \
        --model-path models/medqa/stage2 \
        --manifest data/medqa_corpus/test/medqa_corpus.json \
        --out out/medqa_test/predictions.json

Add --vocoder-dir to also synthesise the spoken answers (slower; only needed
when you want to listen to the Stage II output rather than score the text).
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from inference import OmniModel, units_to_wav   # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocoder-dir", default=None,
                        help="Set to also decode units to wav.")
    parser.add_argument("--wav-dir", default=None)
    parser.add_argument("--prompt-speech", default="llama_omni2/inference/prompt_en.wav")
    parser.add_argument("--hop-len", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(args.manifest) as f:
        records = json.load(f)
    if args.limit:
        records = records[:args.limit]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    omni = OmniModel(args.model_path)

    vocoder = prompt_speech = None
    if args.vocoder_dir:
        import torchaudio
        from llama_omni2.serve.flow_inference import SpeechDecoder
        from cosyvoice.utils.file_utils import load_wav
        vocoder = SpeechDecoder(model_dir=args.vocoder_dir, hop_len=args.hop_len)
        prompt_speech = load_wav(args.prompt_speech, 16000)
        os.makedirs(args.wav_dir or os.path.join(os.path.dirname(args.out), "wav"), exist_ok=True)

    predictions = []
    for n, record in enumerate(records):
        # `question_wavs` is a list because MedQA questions exceed whisper's 30 s window;
        # passing the list feeds them as several <speech> tokens in one turn.
        chunks = record.get("question_wavs") or record["question_wav"]
        started = time.time()
        # Fresh history per item: these are independent questions, not a conversation.
        text, unit_ids = omni.generate(
            [{"role": "user", "content": {"path": chunks}}],
            temperature=args.temperature, top_p=args.top_p,
            max_new_tokens=args.max_new_tokens)
        elapsed = time.time() - started

        wav_path = None
        if vocoder is not None and unit_ids:
            import torchaudio
            speech = units_to_wav(vocoder, unit_ids, prompt_speech)
            if speech is not None:
                wav_dir = args.wav_dir or os.path.join(os.path.dirname(args.out), "wav")
                wav_path = os.path.join(wav_dir, f"{record['id']}.wav")
                torchaudio.save(wav_path, speech.cpu(), 24000, format="wav")

        predictions.append({
            "id": record["id"],
            "answer_idx": record.get("answer_idx"),
            "text": text,
            "num_units": len(unit_ids),
            "wav": wav_path,
            "seconds": round(elapsed, 2),
        })
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(records)}")
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"wrote {len(predictions)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
