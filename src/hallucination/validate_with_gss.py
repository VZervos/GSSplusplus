"""Validate FactEval LLM answers via stock GSS + post-hoc claim checks."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Windows consoles often use cp1252; GSS logs may include Unicode entity names.
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
    except Exception:
        pass

os.environ.setdefault("DATASET", "dbpedia")
os.environ.setdefault("LLM_PROVIDER", "auto")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from hallucination.validation import validate_with_summary
from pipeline.claim_validation import VALID_MODES, run_stock_gss


def _serialize_triple(t: dict) -> dict:
    return {
        "s": t.get("s"),
        "p": t.get("p"),
        "o": t.get("o"),
        "score": t.get("final_score", t.get("score")),
        "importance": t.get("importance"),
        "similarity": t.get("similarity"),
    }


def _default_output(mode: str, output_dir: Path | None = None) -> str:
    base = output_dir if output_dir is not None else (_ROOT / "out" / "hallucination")
    return str(base / f"validation_results_{mode}.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate FactEval LLM answers with stock GSS. "
            "Modes only change the query string; GSS algorithm is unmodified."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        default=str(_ROOT / "out" / "hallucination" / "llm_answers.json"),
        help="LLM answers JSON from answer_facteval.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (default: out/hallucination/validation_results_<mode>.json)",
    )
    parser.add_argument("-n", type=int, default=None, help="Optional limit")
    parser.add_argument(
        "--mode",
        choices=list(VALID_MODES),
        default="subject",
        help="Query mode: answer | subject | claim (default: subject)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(VALID_MODES),
        default=None,
        help="Run several modes sequentially (one output file per mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for validation_results_<mode>.json (default: out/hallucination)",
    )
    args = parser.parse_args()

    modes = args.modes or [args.mode]
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, encoding="utf-8") as f:
        payload = json.load(f)

    answers = payload.get("answers", [])
    if args.n is not None:
        answers = answers[: args.n]

    for mode in modes:
        out_path = args.output
        if out_path is None or len(modes) > 1:
            out_path = _default_output(mode, output_dir)

        results = []
        counts = {
            "decision": {
                "supported": 0,
                "weak_support": 0,
                "unsupported": 0,
                "uncertain": 0,
            },
            "detection": {
                "not_validated": 0,
                "true_support": 0,
                "true_support_granularity": 0,
                "weak_true_support": 0,
                "missed_support": 0,
                "false_support": 0,
                "weak_false_support": 0,
                "uncertain": 0,
            },
            "gold_match": 0,
            "gold_acceptable": 0,
            "granularity_match": 0,
        }

        print("#" * 60)
        print(f"MODE = {mode}  (stock GSS pipeline)")
        print("#" * 60)

        for idx, item in enumerate(answers, 1):
            qid = item["id"]
            question = item["question"]
            llm_answer = item.get("llm_answer") or ""
            relation = item.get("relation")
            print("=" * 60)
            print(f"[{idx}/{len(answers)}] {qid}")
            print(f"Q: {question}")
            print(f"LLM: {llm_answer}")

            triples = []
            meta = {"mode": mode, "relation": relation}
            error = None
            try:
                triples, _, meta = run_stock_gss(
                    mode=mode,
                    question=question,
                    llm_answer=llm_answer,
                    subject=item.get("subject"),
                    relation=relation,
                )
            except Exception as e:
                error = str(e)
                meta = {
                    "mode": mode,
                    "error": error,
                    "answer_entity": None,
                    "answer_uri": None,
                    "subject_uri": None,
                    "subject_label": (item.get("subject") or {}).get("label"),
                    "relation": relation,
                    "gss_query": None,
                    "claim_text": None,
                }
                print(f"PIPELINE ERROR: {error}")

            verdict = validate_with_summary(
                question=question,
                llm_answer=llm_answer,
                gold_answers=item.get("gold_answers") or [],
                subject=item.get("subject"),
                triples=triples,
                meta=meta,
                mode=mode,
                relation=relation,
            )
            print(
                f"Decision={verdict['decision']} detection={verdict['detection']} "
                f"reason={verdict['reason']} triples={verdict['n_triples']} "
                f"gold_match={verdict['gold_match']} gran={verdict.get('granularity_type')}"
            )
            print(f"GSS query: {verdict.get('gss_query')}")
            if verdict.get("supporting_predicates"):
                print(f"Supporting predicates: {verdict['supporting_predicates']}")
            if verdict.get("decision") == "weak_support" and verdict.get("linked_predicates"):
                print(f"Linked (non-allowlist) predicates: {verdict['linked_predicates']}")

            counts["decision"][verdict["decision"]] = (
                counts["decision"].get(verdict["decision"], 0) + 1
            )
            counts["detection"][verdict["detection"]] = (
                counts["detection"].get(verdict["detection"], 0) + 1
            )
            if verdict["gold_match"]:
                counts["gold_match"] += 1
            if verdict.get("gold_acceptable"):
                counts["gold_acceptable"] += 1
            if verdict.get("granularity_match"):
                counts["granularity_match"] += 1

            results.append(
                {
                    "id": qid,
                    "mode": mode,
                    "question": question,
                    "llm_answer": llm_answer,
                    "gold_answers": item.get("gold_answers"),
                    "subject": item.get("subject"),
                    "relation": relation,
                    "gss_query": meta.get("gss_query"),
                    "claim_text": meta.get("claim_text"),
                    "validation": verdict,
                    "subgraph_triples": [_serialize_triple(t) for t in triples[:50]],
                    "pipeline_error": error,
                }
            )
            # Wikidata public endpoint is rate-limited; pause between items.
            if os.environ.get("DATASET", "dbpedia").lower() == "wikidata":
                time.sleep(3.0)

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out_payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input": str(Path(args.input).resolve()),
            "dataset": os.environ.get("DATASET", "dbpedia"),
            "mode": mode,
            "gss": "stock_pipeline",
            "n": len(results),
            "summary": counts,
            "results": results,
        }
        out.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("=" * 60)
        print(f"MODE {mode} summary:", json.dumps(counts, indent=2))
        print(f"Saved validation results to {out}")


if __name__ == "__main__":
    main()
