import json

INPUT_JSON = "./data/qald_9_plus_test_dbpedia.json"
OUTPUT_JSON = "./data/gt_dbpedia_answers.json"


def get_english_question(question_block):
    for q in question_block:
        if q.get("language") == "en":
            return q.get("string")
    return None


def extract_answers(answers_block):
    values = []

    for answer in answers_block:
        results = answer.get("results")
        if not results:
            continue

        bindings = results.get("bindings", [])
        for binding in bindings:
            for var in binding.values():
                values.append(var["value"])

    return values


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    ground_truth = {}

    for entry in data["questions"]:
        qx = f"q{entry['id']}"
        question_en = get_english_question(entry["question"])

        if question_en is None:
            continue

        ground_truth[qx] = {
            "question_id": qx,
            "question": question_en,
            "answers": extract_answers(entry["answers"])
        }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(ground_truth)} questions to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
