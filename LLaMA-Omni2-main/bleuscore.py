
REFERENCE_FILE = "/DATA/carer.ai/LLaMA-Omni2-main/medqa_no_noise/transcript_medqa.json"
MEDQA_DIR = "/DATA/carer.ai/LLaMA-Omni2-main/out"

import json
import os
import re


START = 62
END = 80


# ------------------------------------------------------------
# Extract answer choice: A / B / C / D / E
# ------------------------------------------------------------

def extract_answer(text):
    if not text:
        return None

    # Normalize text
    text = text.upper()
    text = re.sub(r'\s+', ' ', text).strip()

    # ---------------------------------------------------------
    # Strong patterns - highest priority
    # ---------------------------------------------------------

    patterns = [
        # "correct answer is D"
        r'\bCORRECT\s+ANSWER\s+IS\s*[:\-]?\s*\(?([A-E])\)?\b',

        # "answer is D"
        r'\bANSWER\s+IS\s*[:\-]?\s*\(?([A-E])\)?\b',

        # "the answer is D"
        r'\bTHE\s+ANSWER\s+IS\s*[:\-]?\s*\(?([A-E])\)?\b',

        # "answer D"
        r'\bANSWER\s*[:\-]?\s*\(?([A-E])\)?\b',

        # "option D"
        r'\bOPTION\s*[:\-]?\s*\(?([A-E])\)?\b',

        # "choice D"
        r'\bCHOICE\s*[:\-]?\s*\(?([A-E])\)?\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(1)

    return None

# ------------------------------------------------------------
# Load true/reference answers
# ------------------------------------------------------------

with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
    reference_data = json.load(f)


correct = 0
total = 0


# ------------------------------------------------------------
# Evaluate every generated transcript
# ------------------------------------------------------------

for i in range(START, END + 1):

    filename = f"zero_shot_{i}.wav"

    if filename not in reference_data:
        print(f"[SKIP] Reference missing: {filename}")
        continue

    reference_text = reference_data[filename]

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

    # Extract A/B/C/D/E
    reference_answer = extract_answer(reference_text)
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


# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

if total > 0:

    accuracy = correct / total

    print("\n" + "=" * 50)
    print(f"Total evaluated : {total}")
    print(f"Correct         : {correct}")
    print(f"Wrong           : {total - correct}")
    print(f"Answer Accuracy : {accuracy:.4f}")
    print(f"Accuracy %      : {accuracy * 100:.2f}%")
    print("=" * 50)
