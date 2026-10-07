"""Minimal CoLoTa demo: LLM-only vs gold triples vs RAG.

Usage (from repo root):

    python src/colota_demo/run.py
    python src/colota_demo/run.py --k 5 --ks 1,3,5,10 --model llama3.1:8b
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE / "scripts", HERE / "experiments", HERE):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import gold_context
import llm_only
import rag
from dataset import (
    DEMO_IDS,
    ROOT,
    build_corpus,
    gold_label,
    item_triples,
    load_qa,
    select_items,
)
from llm_client import OLLAMA_MODEL
from retriever import DEFAULT_KS, TripleRetriever, ranking_metrics

OUT_DIR = ROOT / "out" / "colota_demo"


def classify_failures(row: dict, k: int) -> list[str]:
    tags: list[str] = []
    gold = row["gold"]
    rag_k = row["rag"][k]
    recall_k = row["recall"][k]

    if row["llm"] != gold and row["gold_triples"] == gold:
        tags.append("knowledge")
    if row["gold_triples"] != gold:
        tags.append("oracle_reasoning")
    if recall_k < 1.0:
        tags.append("retrieval")
    if recall_k >= 1.0 and rag_k != gold:
        tags.append("reasoning")
    if rag_k == "INSUFFICIENT":
        tags.append("insufficient")
    return tags


def accuracy(rows: list[dict], field: str) -> float:
    if not rows:
        return 0.0
    return sum(row[field] == row["gold"] for row in rows) / len(rows)


def accuracy_at(rows: list[dict], family: str, k: int) -> float:
    if not rows:
        return 0.0
    return sum(row[family][k] == row["gold"] for row in rows) / len(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def share(values: list[bool]) -> float:
    return mean([1.0 if v else 0.0 for v in values])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_ks(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return DEFAULT_KS
    ks = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not ks:
        raise ValueError("--ks must list at least one integer")
    return ks


def run(args: argparse.Namespace) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ks = parse_ks(args.ks)
    primary_k = args.k
    if primary_k not in ks:
        ks = tuple(sorted({*ks, primary_k}))

    data = load_qa(Path(args.data) if args.data else None)
    ids = [part.strip() for part in args.ids.split(",") if part.strip()] if args.ids else DEMO_IDS
    examples = select_items(data, ids)
    corpus = build_corpus(examples)
    retriever = TripleRetriever(corpus, model_name=args.embed_model)

    print(f"Using {len(examples)} hand-picked QA items: {', '.join(ids)}")
    print(f"Pooled corpus: {len(corpus)} unique triples")
    print(f"LLM: {args.model}  retriever: {args.embed_model}")
    print(f"Primary k={primary_k}  depth ks={ks}")
    print()

    rows: list[dict] = []
    for i, item in enumerate(examples, start=1):
        qid = item["id"]
        question = item["query"]
        gold = gold_label(item)
        triples = item_triples(item)

        print(f"[{i}/{len(examples)}] {qid}: {question}")

        llm_pred, llm_raw = llm_only.predict(question, model=args.model)
        gt_pred, gt_raw = gold_context.predict(question, triples, model=args.model)

        ranking = retriever.retrieve(question)
        rank_stats = ranking_metrics(triples, ranking, ks=ks)

        rag_preds: dict[int, str] = {}
        rag_raws: dict[int, str] = {}
        for k in ks:
            rag_preds[k], rag_raws[k] = rag.predict(
                question, ranking[:k], model=args.model
            )

        row = {
            "question_id": qid,
            "question": question,
            "gold": gold,
            "llm": llm_pred,
            "gold_triples": gt_pred,
            "rag": rag_preds,
            "recall": rank_stats["recall"],
            "mrr": rank_stats["mrr"],
            "mean_rr": rank_stats["mean_rr"],
            "first_relevant_rank": rank_stats["first_relevant_rank"],
            "gold_ranks": rank_stats["gold_ranks"],
            "n_gold_triples": len(triples),
            "gold_triple_list": triples,
            "ranking": ranking,
            "llm_raw": llm_raw,
            "gold_triples_raw": gt_raw,
            "rag_raw": rag_raws,
        }
        row["failure_tags"] = ";".join(classify_failures(row, primary_k))
        rows.append(row)

        rec_bits = " ".join(f"@{k}={rank_stats['recall'][k]:.2f}" for k in ks)
        rag_bits = " ".join(f"@{k}={rag_preds[k]}" for k in ks)
        print(f"    gold={gold} llm={llm_pred} gold_triples={gt_pred}")
        print(f"    RAG {rag_bits}")
        print(
            f"    MRR={rank_stats['mrr']:.2f}  "
            f"first_rank={rank_stats['first_relevant_rank']}  "
            f"recall {rec_bits}"
        )
        for hit in ranking[:primary_k]:
            marker = "*" if hit["triple"] in triples else " "
            print(f"     {marker} {hit['rank']:>2} {hit['score']:.3f}  {hit['text']}")
        print()

    tag_counts: dict[str, int] = {}
    for row in rows:
        for tag in (row["failure_tags"].split(";") if row["failure_tags"] else []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    first_ranks = [row["first_relevant_rank"] for row in rows]
    known_first = [r for r in first_ranks if r is not None]

    depth = []
    for k in ks:
        depth.append({
            "k": k,
            "recall": mean([row["recall"][k] for row in rows]),
            "rag_accuracy": accuracy_at(rows, "rag", k),
        })

    metrics = {
        "n": len(rows),
        "primary_k": primary_k,
        "ks": list(ks),
        "model": args.model,
        "embed_model": args.embed_model,
        "ids": ids,
        "corpus_size": len(corpus),
        "llm_accuracy": accuracy(rows, "llm"),
        "gold_triples_accuracy": accuracy(rows, "gold_triples"),
        "rag_accuracy": accuracy_at(rows, "rag", primary_k),
        "mrr": mean([row["mrr"] for row in rows]),
        "mean_rr": mean([row["mean_rr"] for row in rows]),
        "mean_first_relevant_rank": mean(known_first),
        "first_relevant_at_1": share([r == 1 for r in known_first]),
        "first_relevant_at_3": share([r is not None and r <= 3 for r in first_ranks]),
        "first_relevant_at_5": share([r is not None and r <= 5 for r in first_ranks]),
        "depth": depth,
        "failure_counts": tag_counts,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(
        json.dumps({"metrics": metrics, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fieldnames = [
        "question_id",
        "gold",
        "llm",
        "gold_triples",
        *[f"rag_{k}" for k in ks],
        *[f"recall_{k}" for k in ks],
        "mrr",
        "mean_rr",
        "first_relevant_rank",
        "failure_tags",
        "question",
    ]
    csv_rows = []
    for row in rows:
        csv_row = {
            "question_id": row["question_id"],
            "gold": row["gold"],
            "llm": row["llm"],
            "gold_triples": row["gold_triples"],
            "mrr": f"{row['mrr']:.4f}",
            "mean_rr": f"{row['mean_rr']:.4f}",
            "first_relevant_rank": row["first_relevant_rank"],
            "failure_tags": row["failure_tags"],
            "question": row["question"],
        }
        for k in ks:
            csv_row[f"rag_{k}"] = row["rag"][k]
            csv_row[f"recall_{k}"] = f"{row['recall'][k]:.4f}"
        csv_rows.append(csv_row)
    write_csv(OUT_DIR / "results.csv", csv_rows, fieldnames)

    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"=== main conditions (k={primary_k}) ===")
    print(f"LLM-only:                       {metrics['llm_accuracy']:.2%}")
    print(f"Gold triples:                   {metrics['gold_triples_accuracy']:.2%}")
    print(f"RAG@{primary_k}:                         {metrics['rag_accuracy']:.2%}")
    print()
    print("=== retrieval depth ===")
    print(f"{'k':>4}  {'Recall':>8}  {'RAG acc':>8}")
    for item in depth:
        print(f"{item['k']:>4}  {item['recall']:>8.2%}  {item['rag_accuracy']:>8.2%}")
    print()
    print("=== ranking ===")
    print(f"MRR (first relevant):           {metrics['mrr']:.3f}")
    print(f"Mean RR (all gold triples):     {metrics['mean_rr']:.3f}")
    print(f"Mean first-relevant rank:       {metrics['mean_first_relevant_rank']:.2f}")
    print(f"First relevant @1:              {metrics['first_relevant_at_1']:.2%}")
    print(f"First relevant @3:              {metrics['first_relevant_at_3']:.2%}")
    print(f"First relevant @5:              {metrics['first_relevant_at_5']:.2%}")
    print()
    print(f"Failure tags @{primary_k}: {tag_counts}")
    print(f"Wrote {OUT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="Primary k for the main table")
    parser.add_argument(
        "--ks",
        default="1,3,5,10",
        help="Comma-separated retrieval depths",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated CoLoTa ids (default: the hand-picked DEMO_IDS)",
    )
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument(
        "--embed-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--data", default=None, help="Path to CoLoTa_qa.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
