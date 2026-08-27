import json
import re


def parse_prediction_units(unit_string):
    return [
        int(unit_id)
        for unit_id in re.findall(r"<(\d+)>", unit_string)
    ]


def convert_answers_to_stage1b(input_path, output_path):
    stage1b_data = []

    with open(input_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            answer = json.loads(line)

            stage1b_data.append(
                {
                    "text": answer["prediction"],
                    "units": parse_prediction_units(
                        answer["prediction_units"]
                    ),
                }
            )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            stage1b_data,
            file,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    convert_answers_to_stage1b(
        "examples/answer.json",
        "examples/stage1b_data.json",
    )
