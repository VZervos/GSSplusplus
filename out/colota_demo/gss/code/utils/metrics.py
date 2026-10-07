"""Accuracy, failure tags, and result writers for the CoLoTa demo suite."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ORIGINAL_METRICS = (
    "accuracy",
    "recall",
    "mrr",
    "mean_rr",
    "first_relevant_rank",
    "failures",
)


def accuracy(preds: list[str], golds: list[str]) -> float:
    if not preds:
        return 0.0
    return sum(p == g for p, g in zip(preds, golds)) / len(preds)


def classify_failures(row: dict, k: int, enabled: set[str]) -> list[str]:
    tags: list[str] = []
    gold = row["gold"]

    if "llm_only" in enabled and "gold_triples" in enabled:
        if row["llm"] != gold and row["gold_triples"] == gold:
            tags.append("knowledge")
    if "gold_triples" in enabled and row["gold_triples"] != gold:
        tags.append("oracle_reasoning")
    if "rag_transformers" in enabled:
        rag_k = row["rag"][k]
        recall_k = row["recall"][k]
        if recall_k < 1.0:
            tags.append("retrieval")
        if recall_k >= 1.0 and rag_k != gold:
            tags.append("reasoning")
        if rag_k == "INSUFFICIENT":
            tags.append("insufficient")
    return tags


def write_results(
    out_dir: Path,
    rows: list[dict],
    metrics: dict,
    enabled: set[str],
    wanted: set[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    ks = metrics.get("ks") or []
    fieldnames = ["id", "question", "gold", "n_gold_triples"]
    if "llm_only" in enabled:
        fieldnames.append("llm")
    if "gold_triples" in enabled:
        fieldnames.append("gold_triples")
    if "rag_transformers" in enabled:
        for k in ks:
            fieldnames.append(f"rag@{k}")
            if "recall" in wanted:
                fieldnames.append(f"recall@{k}")
        if "mrr" in wanted:
            fieldnames.append("mrr")
        if "mean_rr" in wanted:
            fieldnames.append("mean_rr")
        if "first_relevant_rank" in wanted:
            fieldnames.append("first_relevant_rank")
    if "gss" in enabled:
        for k in ks:
            fieldnames.append(f"gss@{k}")
    if "failures" in wanted:
        fieldnames.append("failures")

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            csv_row = {
                "id": row["id"],
                "question": row["question"],
                "gold": row["gold"],
                "n_gold_triples": row["n_gold_triples"],
                "llm": row.get("llm", ""),
                "gold_triples": row.get("gold_triples", ""),
                "mrr": row.get("mrr", ""),
                "mean_rr": row.get("mean_rr", ""),
                "first_relevant_rank": row.get("first_relevant_rank", ""),
                "failures": "|".join(row.get("failures") or []),
            }
            for k in ks:
                csv_row[f"rag@{k}"] = (row.get("rag") or {}).get(k, "")
                csv_row[f"recall@{k}"] = (row.get("recall") or {}).get(k, "")
                csv_row[f"gss@{k}"] = (row.get("gss") or {}).get(k, "")
            writer.writerow(csv_row)


def summarize(
    rows: list[dict],
    enabled: set[str],
    ks: list[int],
    primary_k: int,
    wanted: set[str],
) -> dict:
    golds = [r["gold"] for r in rows]
    metrics: dict = {
        "n": len(rows),
        "approaches": sorted(enabled),
        "ks": ks,
        "primary_k": primary_k,
        "metrics": sorted(wanted),
    }
    if "accuracy" in wanted:
        if "llm_only" in enabled:
            metrics["llm_accuracy"] = accuracy([r["llm"] for r in rows], golds)
        if "gold_triples" in enabled:
            metrics["gold_triples_accuracy"] = accuracy(
                [r["gold_triples"] for r in rows], golds
            )
        if "rag_transformers" in enabled:
            metrics["rag_accuracy"] = accuracy(
                [r["rag"][primary_k] for r in rows], golds
            )
            metrics["rag_accuracy_at_k"] = {
                k: accuracy([r["rag"][k] for r in rows], golds) for k in ks
            }
        if "gss" in enabled:
            metrics["gss_accuracy"] = accuracy(
                [r["gss"][primary_k] for r in rows], golds
            )
            metrics["gss_accuracy_at_k"] = {
                k: accuracy([r["gss"][k] for r in rows], golds) for k in ks
            }
    if "rag_transformers" in enabled:
        if "recall" in wanted:
            metrics["recall"] = {
                k: sum(r["recall"][k] for r in rows) / len(rows) for k in ks
            }
        if "mrr" in wanted:
            metrics["mrr"] = sum(r["mrr"] for r in rows) / len(rows)
        if "mean_rr" in wanted:
            metrics["mean_rr"] = sum(r["mean_rr"] for r in rows) / len(rows)
        if "first_relevant_rank" in wanted:
            ranks = [r.get("first_relevant_rank") for r in rows]
            present = [rank for rank in ranks if rank is not None]
            metrics["mean_first_relevant_rank"] = (
                sum(present) / len(present) if present else None
            )
            metrics["first_relevant_at_k"] = {
                k: sum(
                    (r.get("first_relevant_rank") or 10**9) <= k for r in rows
                )
                / len(rows)
                for k in ks
            }
    if "failures" in wanted:
        metrics["failure_counts"] = dict(
            Counter(tag for r in rows for tag in r.get("failures") or [])
        )
    return metrics


def print_report(
    metrics: dict,
    enabled: set[str],
    primary_k: int,
    wanted: set[str],
) -> None:
    n = metrics["n"]
    print(f"\n=== CoLoTa demo (n={n}) ===")
    if "accuracy" in wanted:
        if "llm_only" in enabled:
            print(f"LLM-only accuracy:            {metrics['llm_accuracy']:.1%}")
        if "gold_triples" in enabled:
            print(f"Gold triples accuracy:        {metrics['gold_triples_accuracy']:.1%}")
        if "rag_transformers" in enabled:
            print(f"RAG@{primary_k} accuracy:               {metrics['rag_accuracy']:.1%}")
        if "gss" in enabled:
            print(f"GSS@{primary_k} accuracy:               {metrics['gss_accuracy']:.1%}")
    if "rag_transformers" in enabled:
        if "mrr" in wanted:
            print(f"MRR (first relevant):         {metrics['mrr']:.3f}")
        if "mean_rr" in wanted:
            print(f"Mean RR (all gold triples):   {metrics['mean_rr']:.3f}")
        if "first_relevant_rank" in wanted:
            mean_rank = metrics.get("mean_first_relevant_rank")
            if mean_rank is not None:
                print(f"Mean first relevant rank:     {mean_rank:.3f}")
        if "recall" in wanted or "accuracy" in wanted:
            print("Recall@k / RAG accuracy@k:")
            for k in metrics["ks"]:
                parts = [f"k={k:<3}"]
                if "recall" in wanted:
                    parts.append(f"recall={metrics['recall'][k]:.1%}")
                if "accuracy" in wanted:
                    parts.append(f"rag_acc={metrics['rag_accuracy_at_k'][k]:.1%}")
                print("  " + "  ".join(parts))
    if "gss" in enabled and "accuracy" in wanted:
        print("GSS accuracy@k:")
        for k in metrics["ks"]:
            print(f"  k={k:<3}  gss_acc={metrics['gss_accuracy_at_k'][k]:.1%}")
    if "failures" in wanted:
        print("Failure tags:", metrics.get("failure_counts", {}))
