import re
import json

INPUT_FILE = "output.txt"
OUTPUT_FILE = "transcripts_51_83.json"

transcripts = {}

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

current_file = None

for line in lines:
    # Find audio filename
    match = re.search(r"\[turn 0\] in=.*?/zero_shot_(\d+)\.wav", line)

    if match:
        number = int(match.group(1))

        if 51 <= number <= 83:
            current_file = f"zero_shot_{number}.wav"
        else:
            current_file = None

        continue

    # Find transcript
    if current_file:
        match = re.search(r"\[turn 0\] text=(.*)", line)

        if match:
            text = match.group(1).strip()

            transcripts[current_file] = text
            current_file = None


# Save JSON
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(transcripts, f, indent=4, ensure_ascii=False)

print(f"Saved {len(transcripts)} transcripts to {OUTPUT_FILE}")

# Show missing files
expected = {
    f"zero_shot_{i}.wav"
    for i in range(51, 84)
}

missing = sorted(expected - set(transcripts))

if missing:
    print("\nMissing transcripts:")
    for filename in missing:
        print(filename)
else:
    print("All transcripts from 51 to 83 found!")
