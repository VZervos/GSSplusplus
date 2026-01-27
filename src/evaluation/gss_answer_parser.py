import csv
import json
import os

INPUT_DIR = "../../out"  # folder that contains q1/, q2/, q4/, etc.
OUTPUT_JSON = "./data/gss_dbpedia_answers.json"


def parse_question_file(q_id: str, file_path: str):
    """
    Reads a qX.txt file and returns structured data
    """
    answers = []

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            answers.append({
                "triple": row["triple"].strip(),
                "importance": float(row["importance"]),
                "similarity": float(row["similarity"]),
                "final": float(row["final"])
            })

    return {
        "question_id": q_id,
        "answers": answers
    }


def main():
    dataset = {}

    for item in sorted(os.listdir(INPUT_DIR)):
        q_dir = os.path.join(INPUT_DIR, item)

        if not os.path.isdir(q_dir):
            continue

        q_id = item
        txt_path = os.path.join(q_dir, f"{q_id}.txt")

        if not os.path.exists(txt_path):
            print(f"Skipping {q_id}: no {q_id}.txt found")
            continue

        dataset[q_id] = parse_question_file(q_id, txt_path)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(dataset)} questions to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
