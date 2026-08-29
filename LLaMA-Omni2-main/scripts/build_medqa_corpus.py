"""Build a MedQA speech corpus: question audio, gold answer text, gold speech units.

Replaces CosyVoice/run_for_medqa.py, which sampled 50 rows from dev.jsonl with no
seed, hardcoded the file index, and wrote invalid JSON (it dumped `""` and then
appended bare objects).

Each row becomes one manifest entry with everything the three training stages
need, so they cannot drift apart:

    Stage I(a)  <question_wav, text>          -- moves answer accuracy
    Stage I(b)  <text, units>                 -- MTTS on medical vocabulary
    Stage II    <question_wav, text, units>   -- gate fusion

The unit targets come from CosyVoice 2's own speech tokenizer
(`speech_tokenizer_v2.onnx`, FSQ 3^8 = 6561 codes at 25 Hz), which is exactly the
`unit_vocab_size: 6561` the model's `<0>..<6560>` tokens index -- so units
extracted here are directly usable as training targets and directly decodable
back to audio by the CosyVoice decoder.

Answer text is built from MedQA's own `answer_idx` and `answer` fields.  Note
that MedQA carries **no** written explanation for its answers (`meta_info` is
just "step1"/"step2&3"), so none is invented here: the target is the letter and
the option text, formatted so bleuscore.py's `^\\s*([A-E])[.,]\\s*` pattern
matches.

Run in the CosyVoice environment (needs onnxruntime); paths are resolved from
this file's location, so the working directory does not matter:

    /DATA/hnc/conda_envs/cosyvoice/bin/python scripts/build_medqa_corpus.py \
        --split train --num 400 --seed 0 --out-dir data/medqa_corpus/train
"""

import argparse
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # LLaMA-Omni2-main
PROJECT_ROOT = os.path.dirname(REPO_ROOT)                                 # carer.ai

# whisper's pad_or_trim truncates at 30 s, and CosyVoiceFrontEnd._extract_speech_token
# asserts on it outright. MedQA stems are far longer than that -- the median question
# plus options synthesises to ~57 s and only 3.5% of the dataset fits in 30 s -- so
# question audio is split into <=30 s chunks and fed as several <speech> tokens in one
# user turn. `prepare_inputs_labels_for_speech_and_text` already splices one 300-frame
# block per token, so this needs no model change; a 57 s question costs 600 frames,
# well inside tokenizer_model_max_length (4096).
MAX_AUDIO_SECONDS = 30.0
# Leave headroom so rounding never pushes a chunk over the assert in the tokenizer.
CHUNK_SECONDS = 28.0

OPTION_KEYS = ("A", "B", "C", "D", "E")


def question_text(row):
    """The stem plus every option, as one utterance to be read aloud."""
    options = ", ".join(f"Answer {k}: {row[k]}" for k in OPTION_KEYS if k in row)
    return f"{row['question']} {options}"


def answer_text(row, style):
    """Gold target. `letter_only` is the tightest thing bleuscore.py can score."""
    letter = row["answer_idx"]
    if style == "letter_only":
        return f"{letter}."
    return f"{letter}. {row['answer']}."


def load_rows(path, num, seed):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if line:
                rows.append((index, json.loads(line)))
    random.Random(seed).shuffle(rows)
    return rows[:num] if num else rows


def synthesize_chunks(cosyvoice, text, prompt_text, prompt_wav, out_prefix, max_seconds):
    """Zero-shot TTS split into <= `max_seconds` wavs.

    `inference_zero_shot` already yields one piece per normalised sentence, so
    packing those greedily keeps every cut on a sentence boundary. A single
    sentence longer than the limit (MedQA has them) is hard-sliced, which is
    ugly but strictly better than dropping the row.

    Returns (paths, durations, n_hard_slices).
    """
    import torch
    import torchaudio

    rate = cosyvoice.sample_rate
    limit = int(max_seconds * rate)

    pieces, hard_slices = [], 0
    for chunk in cosyvoice.inference_zero_shot(text, prompt_text, prompt_wav):
        audio = chunk["tts_speech"]
        while audio.shape[1] > limit:
            pieces.append(audio[:, :limit])
            audio = audio[:, limit:]
            hard_slices += 1
        if audio.shape[1]:
            pieces.append(audio)

    groups, current, current_len = [], [], 0
    for piece in pieces:
        if current and current_len + piece.shape[1] > limit:
            groups.append(current)
            current, current_len = [], 0
        current.append(piece)
        current_len += piece.shape[1]
    if current:
        groups.append(current)

    paths, durations = [], []
    for index, group in enumerate(groups):
        audio = torch.cat(group, dim=1)
        path = f"{out_prefix}_{index:02d}.wav"
        torchaudio.save(path, audio, rate)
        paths.append(path)
        durations.append(audio.shape[1] / rate)
    return paths, durations, hard_slices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--questions", default=None,
                        help="Defaults to data/medqa/data/questions/en_5/<split>.jsonl")
    parser.add_argument("--num", type=int, default=400, help="0 for the whole split.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None,
                        help="Defaults to data/medqa_corpus/<split>")
    parser.add_argument("--cosyvoice-dir", default=os.path.join(PROJECT_ROOT, "CosyVoice"))
    parser.add_argument("--model-dir", default="pretrained_models/CosyVoice2-0.5B",
                        help="Relative to --cosyvoice-dir.")
    parser.add_argument("--prompt-wav", default="asset/zero_shot_prompt.wav",
                        help="Speaker prompt, relative to --cosyvoice-dir. Change this "
                             "(and --prompt-text) to change the accent.")
    parser.add_argument("--prompt-text", default="希望你以后能够做的比我还好呦。")
    parser.add_argument("--answer-style", default="letter_and_text",
                        choices=["letter_and_text", "letter_only"])
    parser.add_argument("--chunk-seconds", type=float, default=CHUNK_SECONDS,
                        help="Max seconds per question chunk; each becomes one <speech> "
                             "token. Must stay under 30 s.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    questions = args.questions or os.path.join(
        PROJECT_ROOT, "data/medqa/data/questions/en_5", f"{args.split}.jsonl")
    out_dir = os.path.abspath(args.out_dir or os.path.join(
        PROJECT_ROOT, "data/medqa_corpus", args.split))
    wav_dir = os.path.join(out_dir, "wav")
    os.makedirs(wav_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "medqa_corpus.json")

    # CosyVoice resolves third_party/ and pretrained_models/ relative to its own root.
    os.chdir(args.cosyvoice_dir)
    sys.path.insert(0, "third_party/Matcha-TTS")
    sys.path.insert(0, ".")
    from cosyvoice.cli.cosyvoice import CosyVoice2

    cosyvoice = CosyVoice2(args.model_dir)
    rows = load_rows(questions, args.num, args.seed)
    print(f"{len(rows)} rows from {questions} (seed {args.seed}) -> {out_dir}")

    existing = {}
    if os.path.isfile(manifest_path) and not args.overwrite:
        with open(manifest_path) as f:
            existing = {r["id"]: r for r in json.load(f)}
        print(f"resuming: {len(existing)} already built")

    records, too_long, failed, hard_sliced = [], 0, 0, 0
    for n, (index, row) in enumerate(rows):
        record_id = f"medqa_{args.split}_{index:05d}"
        if record_id in existing:
            records.append(existing[record_id])
            continue

        text = answer_text(row, args.answer_style)
        try:
            q_wavs, q_seconds, hard = synthesize_chunks(
                cosyvoice, question_text(row), args.prompt_text, args.prompt_wav,
                os.path.join(wav_dir, f"{record_id}_q"), args.chunk_seconds)
            a_wavs, a_seconds, _ = synthesize_chunks(
                cosyvoice, text, args.prompt_text, args.prompt_wav,
                os.path.join(wav_dir, f"{record_id}_a"), args.chunk_seconds)
        except Exception as error:                      # noqa: BLE001 - report and continue
            print(f"  [{record_id}] synthesis failed: {error}")
            failed += 1
            continue

        hard_sliced += hard

        # The answer is the TTS target and must survive the 30 s tokenizer assert in
        # one piece; a multi-chunk answer would need units concatenated across chunk
        # boundaries, which is not what the decoder was trained on.
        if len(a_wavs) != 1 or a_seconds[0] > MAX_AUDIO_SECONDS:
            print(f"  [{record_id}] dropped: answer audio is {sum(a_seconds):.1f}s "
                  f"in {len(a_wavs)} chunks")
            too_long += 1
            for path in q_wavs + a_wavs:
                if os.path.exists(path):
                    os.remove(path)
            continue

        units, _ = cosyvoice.frontend._extract_speech_token(a_wavs[0])
        records.append({
            "id": record_id,
            "source_index": index,
            "question_wavs": q_wavs,
            "answer_wav": a_wavs[0],
            "answer_idx": row["answer_idx"],
            "answer": row["answer"],
            "text": text,
            "units": units.flatten().tolist(),
            "question_seconds": round(sum(q_seconds), 2),
            "question_chunks": len(q_wavs),
            "answer_seconds": round(a_seconds[0], 2),
        })

        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(rows)} processed, {len(records)} kept")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

    # One valid JSON document, written whole.
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nwrote {len(records)} records -> {manifest_path}")
    print(f"dropped {too_long} (answer audio too long), {failed} synthesis failures, "
          f"{hard_sliced} sentences hard-sliced mid-utterance")
    if records:
        lengths = [len(r["units"]) for r in records]
        chunks = [r["question_chunks"] for r in records]
        seconds = [r["question_seconds"] for r in records]
        print(f"units per answer:  min {min(lengths)}, mean {sum(lengths)//len(lengths)}, "
              f"max {max(lengths)}")
        print(f"question chunks:   min {min(chunks)}, mean {sum(chunks)/len(chunks):.1f}, "
              f"max {max(chunks)}  ({sum(seconds)/len(seconds):.0f}s average)")


if __name__ == "__main__":
    main()
