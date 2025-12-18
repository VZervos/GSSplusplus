import sys
import os
import io
from contextlib import redirect_stdout
from io import StringIO

from pipeline.pipeline import pipeline
from utils.dataset import loadDataset

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


def main() -> None:
    # Create output directory at project root
    # _parent_dir is already the project root (parent of src)
    out_dir = os.path.join(_parent_dir, "out")
    out_dir = os.path.abspath(out_dir)
    
    print("=" * 60)
    print("GEMINI SEMANTIC SUMMARIZER - Dataset Processing")
    print("=" * 60)
    print()
    
    dataset_path = os.path.join(os.path.dirname(__file__), "test", "small.json")
    print(f"Loading dataset: {dataset_path}")
    english_questions = loadDataset(dataset_path)
    print(f"Found {len(english_questions)} English questions")
    print(f"Output directory: {out_dir}")
    print()
    
    for idx, q_data in enumerate(english_questions, 1):
        question_id = q_data["id"]
        question_text = q_data["question"]

        print(f"\nQuestion {idx}/{len(english_questions)} (ID: {question_id})")
        print(f"  {question_text}")
        print()
        
        # Create question-specific output directory
        question_dir = os.path.join(out_dir, f"q{question_id}")
        os.makedirs(question_dir, exist_ok=True)
        
        # Capture console output for this question
        log_buffer = StringIO()
        
        try:
            # Redirect stdout to capture pipeline output
            with redirect_stdout(log_buffer):
                print(f"Question {idx}/{len(english_questions)} (ID: {question_id})")
                print(f"  {question_text}")
                print()
                
                top_triples, uri_importance_map = pipeline(question_text)

                print(f"\nResults for question {question_id}:")
                print(f"  {len(top_triples)} triples found")
                print(f"  {len(uri_importance_map)} URIs with importance scores")
                
                # Print the final list of triples with importance scores
                for triple in top_triples:
                    s = triple.get('s', '')
                    p = triple.get('p', '')
                    o = triple.get('o', '')

                    importance = triple.get('importance', 0.0)
                    similarity = triple.get('similarity', 0.0)
                    final_score = triple.get('final_score', 0.0)

                    print(
                        f"  {s} {p} {o} "
                        f"| importance={importance:.4f} "
                        f"| similarity={similarity:.4f} "
                        f"| final={final_score:.4f}"
                    )
            
            # Get captured output
            log_content = log_buffer.getvalue()
            
            # Write log file
            log_file_path = os.path.join(question_dir, f"log_q{question_id}.txt")
            with open(log_file_path, 'w', encoding='utf-8') as log_file:
                log_file.write(log_content)
            print(f"  Log saved to: {log_file_path}")
            
            # Write results file (triples with scores)
            results_file_path = os.path.join(question_dir, f"q{question_id}.txt")
            with open(results_file_path, 'w', encoding='utf-8') as results_file:
                # Write header
                results_file.write("triple,importance,similarity,final\n")
                
                # Write each triple
                for triple in top_triples:
                    s = triple.get('s', '')
                    p = triple.get('p', '')
                    o = triple.get('o', '')
                    
                    # Format triple as "subject predicate object"
                    triple_str = f"{s} {p} {o}"
                    
                    importance = triple.get('importance', 0.0)
                    similarity = triple.get('similarity', 0.0)
                    final_score = triple.get('final_score', 0.0)
                    
                    # Write CSV line (escape commas in triple if needed)
                    results_file.write(
                        f'"{triple_str}",{importance:.6f},{similarity:.6f},{final_score:.6f}\n'
                    )
            
            print(f"  Results saved to: {results_file_path}")
            print(f"  {len(top_triples)} triples written")
            
        except Exception as e:
            error_msg = f"Error processing question {question_id}: {str(e)}\n"
            print(error_msg)
            
            # Write error to log file
            log_file_path = os.path.join(question_dir, f"log_q{question_id}.txt")
            error_log_content = log_buffer.getvalue() + error_msg
            with open(log_file_path, 'w', encoding='utf-8') as log_file:
                log_file.write(error_log_content)
            
            # Create empty results file for failed questions
            results_file_path = os.path.join(question_dir, f"q{question_id}.txt")
            with open(results_file_path, 'w', encoding='utf-8') as results_file:
                results_file.write("triple,importance,similarity,final\n")
            
            continue

    print("\n" + "=" * 60)
    print("Dataset processing completed!")
    print(f"Results saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
