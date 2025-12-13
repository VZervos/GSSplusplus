import json
import sys


def loadDataset(dataset_path: str):
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at {dataset_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in dataset file: {str(e)}")
        sys.exit(1)

    # Extract English questions
    questions = data.get("questions", [])
    english_questions = []

    for question_item in questions:
        question_id = question_item.get("id", "unknown")
        question_list = question_item.get("question", [])

        # Find the English question
        for q in question_list:
            if q.get("language") == "en":
                english_questions.append({
                    "id": question_id,
                    "question": q.get("string", "")
                })
                break  # Take only the first English question if multiple exist
    return english_questions
