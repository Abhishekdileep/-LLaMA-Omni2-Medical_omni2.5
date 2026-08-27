REFERENCE_FILE = "/DATA/carer.ai/LLaMA-Omni2-main/medqa_in/samples.json"
MEDQA_DIR = "/DATA/carer.ai/LLaMA-Omni2-main/medqa_in"

import json
import os
import re


START = 52
END = 80


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


with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
    reference_data = json.load(f)


correct = 0
total = 0


for i in range(START, END + 1):

    filename = f"zero_shot_{i}.wav"

    # Python lists are 0-indexed.
    reference_index = i - 51
    
    if reference_index >= len(reference_data):
        print(f"[SKIP] Reference missing for {filename}")
        continue
    reference_item = reference_data[reference_index]

    reference_answer = reference_item["answer_idx"]

    transcript_path = os.path.join(
        MEDQA_DIR,
        f"zero_shot_{i}",
        "transcript.json"
    )

    if not os.path.exists(transcript_path):
        print(f"[SKIP] Transcript missing: {transcript_path}")
        continue

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    generated_text = transcript_data[0]["text"]

    generated_answer = extract_answer(generated_text)

    total += 1

    if reference_answer == generated_answer:
        correct += 1
        result = "CORRECT"
    else:
        result = "WRONG"

    print(
        f"{filename:<20} "
        f"True={reference_answer}  "
        f"Generated={generated_answer}  "
        f"{result}"
    )


if total > 0:

    accuracy = correct / total

    print("\n" + "=" * 50)
    print(f"Total evaluated : {total}")
    print(f"Correct         : {correct}")
    print(f"Wrong           : {total - correct}")
    print(f"Answer Accuracy : {accuracy:.4f}")
    print(f"Accuracy %      : {accuracy * 100:.2f}%")
    print("=" * 50)