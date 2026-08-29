"""Project one MedQA manifest into the three stage-specific data files.

`build_medqa_corpus.py` writes a single manifest holding audio, gold text and
gold units together. Each training stage consumes a different subset of that,
and deriving all three from one source is what keeps them consistent -- a
Stage II example and the Stage I(a) example it depends on must describe the
same utterance, or the cached hidden states line up with the wrong text.

    python scripts/project_manifest.py --manifest data/medqa_corpus/train/medqa_corpus.json
"""

import argparse
import json
import os


def project(manifest_path, out_dir=None):
    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    out_dir = out_dir or os.path.dirname(os.path.abspath(manifest_path))

    # Stage I(a): the questions.json shape SupervisedSpeechDataset expects. `speech` is a
    # list because a MedQA question rarely fits in whisper's 30 s window -- each chunk
    # becomes its own <speech> token in the same user turn.
    stage1a = [{
        "id": r["id"],
        "conversation": [
            {"from": "human", "speech": r["question_wavs"]},
            {"from": "gpt", "text": r["text"]},
        ],
    } for r in records]

    # Stage I(b): text -> units, no audio input.
    stage1b = [{"text": r["text"], "units": r["units"]} for r in records]

    # Stage II: adds the question audio the LLM hidden states are conditioned on.
    stage2 = [{
        "id": r["id"],
        "speech": r["question_wavs"],
        "text": r["text"],
        "units": r["units"],
    } for r in records]

    written = []
    for name, data in (("stage1a_data.json", stage1a),
                       ("stage1b_data.json", stage1b),
                       ("stage2_data.json", stage2)):
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        written.append(path)
        print(f"wrote {len(data):5d} records -> {path}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", default=None,
                        help="Defaults to the manifest's own directory.")
    args = parser.parse_args()
    project(args.manifest, args.out_dir)
