import json
import sys


def loadDataset(dataset_path: str) -> list:
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


def extract_query_keywords(extraction: dict, query: str) -> set:
    """Extracts keywords from extraction and query for similarity computation."""
    query_keywords = set()
    
    # Handle both new format (list of dicts) and legacy format (list of strings)
    for entity in extraction["entities"]:
        if isinstance(entity, dict):
            entity_name = entity.get("name", "")
        else:
            entity_name = entity
        
        query_keywords.add(entity_name.lower())
        query_keywords.update(entity_name.lower().split("_"))
        query_keywords.update(entity_name.lower().replace("_", " ").split())
    
    query_keywords.add(query.lower())
    query_keywords.update(query.lower().split())
    return query_keywords
