"""Score MedQA answer accuracy from generated transcripts.

Two input layouts are supported:

* `--predictions` -- the single JSON file `scripts/eval_medqa.py` writes, joined
  to its manifest by `id`. Preferred: it works for any split and any size.
* `--pred-dir`    -- the older layout, one `<name>/transcript.json` per wav as
  produced by `run.sh`, joined by filename. Kept so existing runs under
  `medqa_in/` can still be scored.

The old script hardcoded START=52, END=80 and an `i - 51` offset into a
50-element samples.json, which only described one specific run.

    python bleuscore.py --predictions out/medqa_test/predictions.json \
                        --manifest data/medqa_corpus/test/medqa_corpus.json

    python bleuscore.py --pred-dir medqa_in --manifest medqa_in/samples.json \
                        --start 52 --end 80 --index-offset 51
"""

import argparse
import json
import os
import re


def extract_answer(text):
    if not text:
        return None

    text = text.upper()
    text = re.sub(r"\s+", " ", text).strip()

    patterns = [
        r"\bCORRECT\s+ANSWER\s+IS\s*[:\-]?\s*\(?([A-E])\)?\b",
        r"\bTHE\s+ANSWER\s+IS\s*[:\-]?\s*\(?([A-E])\)?\b",
        r"\bANSWER\s+IS\s*[:\-]?\s*\(?([A-E])\)?\b",
        r"\bANSWER\s*[:\-]?\s*\(?([A-E])\)?\b",
        r"\bOPTION\s*[:\-]?\s*\(?([A-E])\)?\b",
        r"\bCHOICE\s*[:\-]?\s*\(?([A-E])\)?\b",

        # Output starts with: "A, The pathogen..."
        r"^\s*([A-E])[.,]\s*",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(1)

    return None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pairs_from_predictions(predictions_path, manifest_path):
    """Join eval_medqa.py output to its manifest by id."""
    predictions = load_json(predictions_path)
    gold = {}
    if manifest_path:
        gold = {r["id"]: r["answer_idx"] for r in load_json(manifest_path)}

    for prediction in predictions:
        reference = gold.get(prediction["id"], prediction.get("answer_idx"))
        if reference is None:
            print(f"[SKIP] no gold answer for {prediction['id']}")
            continue
        yield prediction["id"], reference, prediction["text"]


def pairs_from_dir(pred_dir, manifest_path, start, end, index_offset):
    """Older layout: <pred_dir>/zero_shot_<i>/transcript.json against a list manifest."""
    reference_data = load_json(manifest_path)

    for i in range(start, end + 1):
        name = f"zero_shot_{i}"
        reference_index = i - index_offset
        if not 0 <= reference_index < len(reference_data):
            print(f"[SKIP] Reference missing for {name}.wav")
            continue

        transcript_path = os.path.join(pred_dir, name, "transcript.json")
        if not os.path.exists(transcript_path):
            print(f"[SKIP] Transcript missing: {transcript_path}")
            continue

        yield name, reference_data[reference_index]["answer_idx"], load_json(transcript_path)[0]["text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=None,
                        help="JSON written by scripts/eval_medqa.py.")
    parser.add_argument("--pred-dir", default=None,
                        help="Directory of <name>/transcript.json, as run.sh produces.")
    parser.add_argument("--manifest", default=None,
                        help="Corpus manifest (or samples.json) holding answer_idx.")
    parser.add_argument("--start", type=int, default=52, help="--pred-dir mode only.")
    parser.add_argument("--end", type=int, default=80, help="--pred-dir mode only.")
    parser.add_argument("--index-offset", type=int, default=51, help="--pred-dir mode only.")
    parser.add_argument("--quiet", action="store_true", help="Only print the summary.")
    args = parser.parse_args()

    if not args.predictions and not args.pred_dir:
        parser.error("pass --predictions or --pred-dir")
    if args.pred_dir and not args.manifest:
        parser.error("--pred-dir needs --manifest")

    if args.predictions:
        pairs = pairs_from_predictions(args.predictions, args.manifest)
    else:
        pairs = pairs_from_dir(args.pred_dir, args.manifest, args.start, args.end,
                               args.index_offset)

    correct = 0
    total = 0
    unparsed = 0

    for name, reference_answer, generated_text in pairs:
        generated_answer = extract_answer(generated_text)
        total += 1
        if generated_answer is None:
            unparsed += 1

        if reference_answer == generated_answer:
            correct += 1
            result = "CORRECT"
        else:
            result = "WRONG"

        if not args.quiet:
            print(f"{name:<24} True={reference_answer}  "
                  f"Generated={generated_answer}  {result}")

    if total > 0:
        accuracy = correct / total
        print("\n" + "=" * 50)
        print(f"Total evaluated : {total}")
        print(f"Correct         : {correct}")
        print(f"Wrong           : {total - correct}")
        print(f"Unparsed        : {unparsed}  (no A-E found in the output)")
        print(f"Answer Accuracy : {accuracy:.4f}")
        print(f"Accuracy %      : {accuracy * 100:.2f}%")
        print("=" * 50)
    else:
        print("nothing evaluated")


if __name__ == "__main__":
    main()
