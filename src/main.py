import sys
import os
import io

from pipeline.pipeline import pipeline
from utils.dataset import loadDataset

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


def main():
    print("=" * 60)
    print("GEMINI SEMANTIC SUMMARIZER - Dataset Processing")
    print("=" * 60)
    print()
    
    dataset_path = os.path.join(os.path.dirname(__file__), "test", "small.json")
    print(f"Loading dataset: {dataset_path}")
    english_questions = loadDataset(dataset_path)
    print(f"Found {len(english_questions)} English questions")
    print()
    
    for idx, q_data in enumerate(english_questions, 1):
        question_id = q_data["id"]
        question_text = q_data["question"]

        print(f"\nQuestion {idx}/{len(english_questions)} (ID: {question_id})")
        print(f"  {question_text}")
        print()
        
        try:
            result = pipeline(question_text)
            triples = result["triples"]
            importance_map = result["importance_map"]
            
            print(f"\nResults for question {question_id}:")
            print(f"  {len(triples)} triples found")
            print(f"  {len(importance_map)} URIs with importance scores")
            
            # Print final list of triples with importance scores
            if triples:
                print(f"\nFinal triples with importance scores:")
                for triple in triples:
                    s = triple.get('s', '')
                    p = triple.get('p', '')
                    o = triple.get('o', '')
                    importance = triple.get('importance', 0.0)
                    print(f"  {s} {p} {o} -> importance: {importance:.4f}")
            
        except Exception as e:
            print(f"Error processing question {question_id}: {str(e)}")
            continue
    
    print("\n" + "=" * 60)
    print("Dataset processing completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
