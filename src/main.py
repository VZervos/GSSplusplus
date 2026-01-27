import argparse
import io
import os
import sys
from contextlib import redirect_stdout
from io import StringIO

from config.settings import (
    VIS_MAX_NODES,
    OUTPUT_GRAPH_FILENAME,
    LLM_PROVIDER,
    DATASET,
    OUTPUT_BASE_DIR
)
from pipeline.pipeline import pipeline
from utils.dataset import loadDataset
from utils.visualization import visualize_knowledge_graph

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


def parse_arguments():
    """Parse command-line arguments."""
    # Determine default test file based on current dataset setting
    from config.settings import DATASET as current_dataset
    if current_dataset.lower() == "wikidata":
        default_test_file = "small_wd.json"
    else:
        default_test_file = "small_db.json"
    
    default_dataset_path = os.path.join(os.path.dirname(__file__), "test", default_test_file)
    
    parser = argparse.ArgumentParser(
        description='Graph Semantic Summarizer - Process QALD dataset questions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default test file
  python src/main.py
  
  # Specify input file
  python src/main.py -i dataset/QALD_9_plus-main/data/qald_9_plus_test_dbpedia.json
  
  # Specify input file and dataset
  python src/main.py -i dataset/QALD_9_plus-main/data/qald_9_plus_test_wikidata.json --dataset wikidata
  
  # Use relative or absolute paths
  python src/main.py -i ./src/test/small_db.json
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=default_dataset_path,
        help=f'Path to input JSON dataset file (default: {default_dataset_path})'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['dbpedia', 'wikidata'],
        default=None,
        help='Knowledge graph dataset to use: "dbpedia" or "wikidata" (default: from settings.py or environment variable)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for results (default: ./out)'
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    
    # Set dataset if provided via command-line
    if args.dataset:
        os.environ['DATASET'] = args.dataset.lower()
        # Reload settings and dependent modules to pick up the new DATASET value
        import importlib
        from config import settings
        importlib.reload(settings)
        # Reload modules that use DATASET
        from utils import dbpedia, parser
        from evaluation import evaluate
        importlib.reload(dbpedia)
        importlib.reload(parser)
        importlib.reload(evaluate)
        # Re-import to get updated values
        from config.settings import DATASET as current_dataset
    else:
        # Read from settings (which reads from env var or defaults to "dbpedia")
        from config.settings import DATASET as current_dataset
    
    from config.settings import LLM_PROVIDER
    provider_name = LLM_PROVIDER.upper()
    
    # Resolve input file path
    if os.path.isabs(args.input):
        dataset_path = args.input
    else:
        # Try relative to current working directory first
        if os.path.exists(args.input):
            dataset_path = os.path.abspath(args.input)
        else:
            # Try relative to script directory
            dataset_path = os.path.join(os.path.dirname(__file__), args.input)
            if not os.path.exists(dataset_path):
                # Try relative to parent directory
                dataset_path = os.path.join(_parent_dir, args.input)
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Input file not found: {args.input}")
        print(f"  Tried: {dataset_path}")
        sys.exit(1)
    
    dataset_path = os.path.abspath(dataset_path)
    
    # Set output directory
    if args.output:
        out_dir = os.path.abspath(args.output)
    else:
        out_dir = os.path.join(_parent_dir, "out")
        out_dir = os.path.abspath(out_dir)
    
    print("=" * 60)
    print(f"{provider_name} SEMANTIC SUMMARIZER - Dataset Processing")
    print("=" * 60)
    print()
    print(f"Dataset: {current_dataset.upper()}")
    print(f"Input file: {dataset_path}")
    print(f"Output directory: {out_dir}")
    print()
    
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

        question_dir = os.path.join(out_dir, f"q{question_id}")
        os.makedirs(question_dir, exist_ok=True)

        log_buffer = StringIO()

        try:
            with redirect_stdout(log_buffer):
                print(f"Question {idx}/{len(english_questions)} (ID: {question_id})")
                print(f"  {question_text}")
                print()

                top_triples, uri_importance_map = pipeline(question_text)

                print(f"\nResults for question {question_id}:")
                print(f"  {len(top_triples)} triples found")
                print(f"  {len(uri_importance_map)} URIs with importance scores")

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

            log_content = log_buffer.getvalue()

            log_file_path = os.path.join(question_dir, f"log_q{question_id}.txt")
            with open(log_file_path, 'w', encoding='utf-8') as log_file:
                log_file.write(log_content)
            print(f"  Log saved to: {log_file_path}")

            results_file_path = os.path.join(question_dir, f"q{question_id}.txt")
            with open(results_file_path, 'w', encoding='utf-8') as results_file:
                results_file.write("triple,importance,similarity,final\n")

                for triple in top_triples:
                    s = triple.get('s', '')
                    p = triple.get('p', '')
                    o = triple.get('o', '')

                    triple_str = f"{s} {p} {o}"
                    importance = triple.get('importance', 0.0)
                    similarity = triple.get('similarity', 0.0)
                    final_score = triple.get('final_score', 0.0)

                    results_file.write(
                        f'"{triple_str}",{importance:.6f},{similarity:.6f},{final_score:.6f}\n'
                    )

            print(f"  Results saved to: {results_file_path}")
            print(f"  {len(top_triples)} triples written")

            if top_triples:
                try:
                    graph_filename = OUTPUT_GRAPH_FILENAME.format(id=question_id)
                    visualization_path = os.path.join(question_dir, graph_filename)
                    visualize_knowledge_graph(
                        top_triples,
                        visualization_path,
                        uri_importance_map=uri_importance_map,
                        max_nodes=VIS_MAX_NODES
                    )
                except Exception as viz_error:
                    print(f"  Warning: Could not generate visualization: {str(viz_error)}")

        except Exception as e:
            error_msg = f"Error processing question {question_id}: {str(e)}\n"
            print(error_msg)

            log_file_path = os.path.join(question_dir, f"log_q{question_id}.txt")
            error_log_content = log_buffer.getvalue() + error_msg
            with open(log_file_path, 'w', encoding='utf-8') as log_file:
                log_file.write(error_log_content)

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
