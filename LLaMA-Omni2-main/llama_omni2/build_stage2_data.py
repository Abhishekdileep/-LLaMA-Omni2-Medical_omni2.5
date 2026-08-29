"""Join questions.json and answer.json into the Stage II data format.

`convert_answers.py` already turns answer.json into Stage I(b) records
({"text", "units"}).  Stage II additionally needs the *input audio*, because the
gate fusion is conditioned on the LLM hidden states that audio produces.  This
script carries the wav path through by joining on the shared id
(answer.json's `question_id` == questions.json's `id`).

    python llama_omni2/build_stage2_data.py \
        --questions examples/questions.json \
        --answers   examples/answer.json \
        --output    examples/stage2_data.json
"""

import argparse
import json

from llama_omni2.convert_answers import parse_prediction_units


def load_answers(path):
    """answer.json is JSON-lines, one prediction per line."""
    answers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                answers.append(json.loads(line))
    return answers


def build(questions_path, answers_path, output_path):
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # The human turn of each conversation holds the wav the answer was generated from.
    speech_by_id = {}
    for item in questions:
        for turn in item["conversation"]:
            if turn["from"] == "human" and "speech" in turn:
                speech_by_id[item["id"]] = turn["speech"]
                break

    records, orphans = [], []
    for answer in load_answers(answers_path):
        qid = answer["question_id"]
        if qid not in speech_by_id:
            orphans.append(qid)
            continue
        records.append({
            "id": qid,
            "speech": speech_by_id[qid],
            "text": answer["prediction"],
            "units": parse_prediction_units(answer["prediction_units"]),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"wrote {len(records)} records -> {output_path}")
    if orphans:
        print(f"dropped {len(orphans)} answers with no matching question id "
              f"(first: {orphans[:3]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="examples/questions.json")
    parser.add_argument("--answers", default="examples/answer.json")
    parser.add_argument("--output", default="examples/stage2_data.json")
    args = parser.parse_args()
    build(args.questions, args.answers, args.output)
