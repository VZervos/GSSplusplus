import sys
import os
import io

from pipeline.pipeline import pipeline
from utils.dataset import loadDataset

# Set UTF-8 encoding for stdout on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path to allow imports to work from both project root and src directory
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


def main():
    print("=" * 60)
    print("GEMINI SEMANTIC SUMMARIZER - Dataset Processing")
    print("=" * 60)
    print()
    
    # Load the small.json dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "test", "small.json")
    english_questions = loadDataset(dataset_path)
    print(f"Found {len(english_questions)} English questions in the dataset")
    print()
    
    # Process each question
    for idx, q_data in enumerate(english_questions, 1):
        question_id = q_data["id"]
        question_text = q_data["question"]

        print(f"Question {idx}/{len(english_questions)} (ID: {question_id}): {question_text}")
        print()
        
        try:
            triples = pipeline(question_text)
            print()
            print(f"{len(triples)} triples were found for question {question_id}:")
            for triple in triples:
                print(f"  {triple['s']} {triple['p']} {triple['o']}.")
            
        except Exception as e:
            print(f"ERROR processing question {question_id}:")
            print(str(e))
            print()
            continue
    
    print("=" * 60)
    print("Dataset processing completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
