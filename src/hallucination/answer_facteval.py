"""Ask a local/remote LLM FactEval T-REx questions and save answers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from services.llm_client import call_llm_api, extract_response_text


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ANSWER_PROMPT = """You are a factual question-answering system.
Answer the question with a short factual answer only.
Prefer a single entity name (person, place, organization, occupation, etc.).
Do not explain. Do not use full sentences unless necessary.

Question: {question}
Answer:"""


def answer_question(question: str) -> str:
    response = call_llm_api(ANSWER_PROMPT.format(question=question), timeout=180)
    text = extract_response_text(response).strip()
    # Keep first non-empty line; strip quotes/bullets
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-* ").strip().strip('"').strip("'")
        if cleaned:
            return cleaned
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer FactEval T-REx questions with an LLM")
    parser.add_argument(
        "-i",
        "--input",
        default=str(_repo_root() / "dataset" / "facteval_trex" / "sample_10.json"),
        help="FactEval sample JSON",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=str(_repo_root() / "out" / "hallucination" / "llm_answers.json"),
        help="Where to save LLM answers",
    )
    parser.add_argument("-n", type=int, default=None, help="Optional limit on number of questions")
    args = parser.parse_args()

    os.environ.setdefault("LLM_PROVIDER", "auto")

    with open(args.input, encoding="utf-8") as f:
        sample = json.load(f)

    questions = sample.get("questions", [])
    if args.n is not None:
        questions = questions[: args.n]

    results = []
    for idx, item in enumerate(questions, 1):
        qid = item["id"]
        question = item["question"]
        gold = item.get("gold_answers", [])
        print("=" * 60)
        print(f"[{idx}/{len(questions)}] {qid}")
        print(f"Q: {question}")
        try:
            llm_answer = answer_question(question)
            err = None
        except Exception as e:
            llm_answer = ""
            err = str(e)
            print(f"ERROR: {err}")
        print(f"A: {llm_answer}")
        print(f"Gold: {[g['label'] for g in gold]}")
        results.append(
            {
                "id": qid,
                "question": question,
                "llm_answer": llm_answer,
                "gold_answers": gold,
                "subject": item.get("subject"),
                "relation": item.get("relation"),
                "triple": item.get("triple"),
                "error": err,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(args.input).resolve()),
        "n": len(results),
        "answers": results,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 60)
    print(f"Saved {len(results)} answers to {out}")


if __name__ == "__main__":
    main()
