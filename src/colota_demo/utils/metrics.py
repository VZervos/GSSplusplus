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
            if "recall" in wanted:
                fieldnames.append(f"gss_recall@{k}")
        if "mrr" in wanted:
            fieldnames.append("gss_mrr")
        if "mean_rr" in wanted:
            fieldnames.append("gss_mean_rr")
        if "first_relevant_rank" in wanted:
            fieldnames.append("gss_first_relevant_rank")
    if "isummary" in enabled:
        for k in ks:
            fieldnames.append(f"isummary@{k}")
            if "recall" in wanted:
                fieldnames.append(f"isummary_recall@{k}")
        if "mrr" in wanted:
            fieldnames.append("isummary_mrr")
        if "mean_rr" in wanted:
            fieldnames.append("isummary_mean_rr")
        if "first_relevant_rank" in wanted:
            fieldnames.append("isummary_first_relevant_rank")
    if "ppr" in enabled:
        for k in ks:
            fieldnames.append(f"ppr@{k}")
            if "recall" in wanted:
                fieldnames.append(f"ppr_recall@{k}")
        if "mrr" in wanted:
            fieldnames.append("ppr_mrr")
        if "mean_rr" in wanted:
            fieldnames.append("ppr_mean_rr")
        if "first_relevant_rank" in wanted:
            fieldnames.append("ppr_first_relevant_rank")
    if "extract_embed" in enabled:
        for k in ks:
            fieldnames.append(f"extract_embed@{k}")
            if "recall" in wanted:
                fieldnames.append(f"extract_embed_recall@{k}")
        if "mrr" in wanted:
            fieldnames.append("extract_embed_mrr")
        if "mean_rr" in wanted:
            fieldnames.append("extract_embed_mean_rr")
        if "first_relevant_rank" in wanted:
            fieldnames.append("extract_embed_first_relevant_rank")
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
                csv_row[f"gss_recall@{k}"] = (row.get("gss_recall") or {}).get(k, "")
                csv_row[f"isummary@{k}"] = (row.get("isummary") or {}).get(k, "")
                csv_row[f"isummary_recall@{k}"] = (
                    (row.get("isummary_recall") or {}).get(k, "")
                )
                csv_row[f"ppr@{k}"] = (row.get("ppr") or {}).get(k, "")
                csv_row[f"ppr_recall@{k}"] = (row.get("ppr_recall") or {}).get(k, "")
                csv_row[f"extract_embed@{k}"] = (row.get("extract_embed") or {}).get(
                    k, ""
                )
                csv_row[f"extract_embed_recall@{k}"] = (
                    (row.get("extract_embed_recall") or {}).get(k, "")
                )
            csv_row["gss_mrr"] = row.get("gss_mrr", "")
            csv_row["gss_mean_rr"] = row.get("gss_mean_rr", "")
            csv_row["gss_first_relevant_rank"] = row.get(
                "gss_first_relevant_rank", ""
            )
            csv_row["isummary_mrr"] = row.get("isummary_mrr", "")
            csv_row["isummary_mean_rr"] = row.get("isummary_mean_rr", "")
            csv_row["isummary_first_relevant_rank"] = row.get(
                "isummary_first_relevant_rank", ""
            )
            csv_row["ppr_mrr"] = row.get("ppr_mrr", "")
            csv_row["ppr_mean_rr"] = row.get("ppr_mean_rr", "")
            csv_row["ppr_first_relevant_rank"] = row.get(
                "ppr_first_relevant_rank", ""
            )
            csv_row["extract_embed_mrr"] = row.get("extract_embed_mrr", "")
            csv_row["extract_embed_mean_rr"] = row.get("extract_embed_mean_rr", "")
            csv_row["extract_embed_first_relevant_rank"] = row.get(
                "extract_embed_first_relevant_rank", ""
            )
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
        if "isummary" in enabled:
            metrics["isummary_accuracy"] = accuracy(
                [r["isummary"][primary_k] for r in rows], golds
            )
            metrics["isummary_accuracy_at_k"] = {
                k: accuracy([r["isummary"][k] for r in rows], golds) for k in ks
            }
        if "ppr" in enabled:
            metrics["ppr_accuracy"] = accuracy(
                [r["ppr"][primary_k] for r in rows], golds
            )
            metrics["ppr_accuracy_at_k"] = {
                k: accuracy([r["ppr"][k] for r in rows], golds) for k in ks
            }
        if "extract_embed" in enabled:
            metrics["extract_embed_accuracy"] = accuracy(
                [r["extract_embed"][primary_k] for r in rows], golds
            )
            metrics["extract_embed_accuracy_at_k"] = {
                k: accuracy([r["extract_embed"][k] for r in rows], golds) for k in ks
            }
    if "gss" in enabled:
        if "recall" in wanted:
            metrics["gss_recall"] = {
                k: sum(r["gss_recall"][k] for r in rows) / len(rows) for k in ks
            }
        if "mrr" in wanted:
            metrics["gss_mrr"] = sum(r["gss_mrr"] for r in rows) / len(rows)
        if "mean_rr" in wanted:
            metrics["gss_mean_rr"] = sum(r["gss_mean_rr"] for r in rows) / len(rows)
        if "first_relevant_rank" in wanted:
            ranks = [r.get("gss_first_relevant_rank") for r in rows]
            present = [rank for rank in ranks if rank is not None]
            metrics["gss_mean_first_relevant_rank"] = (
                sum(present) / len(present) if present else None
            )
            metrics["gss_first_relevant_at_k"] = {
                k: sum(
                    (r.get("gss_first_relevant_rank") or 10**9) <= k for r in rows
                )
                / len(rows)
                for k in ks
            }
    if "isummary" in enabled:
        if "recall" in wanted:
            metrics["isummary_recall"] = {
                k: sum(r["isummary_recall"][k] for r in rows) / len(rows)
                for k in ks
            }
        if "mrr" in wanted:
            metrics["isummary_mrr"] = sum(r["isummary_mrr"] for r in rows) / len(rows)
        if "mean_rr" in wanted:
            metrics["isummary_mean_rr"] = (
                sum(r["isummary_mean_rr"] for r in rows) / len(rows)
            )
        if "first_relevant_rank" in wanted:
            ranks = [r.get("isummary_first_relevant_rank") for r in rows]
            present = [rank for rank in ranks if rank is not None]
            metrics["isummary_mean_first_relevant_rank"] = (
                sum(present) / len(present) if present else None
            )
            metrics["isummary_first_relevant_at_k"] = {
                k: sum(
                    (r.get("isummary_first_relevant_rank") or 10**9) <= k
                    for r in rows
                )
                / len(rows)
                for k in ks
            }
    if "ppr" in enabled:
        if "recall" in wanted:
            metrics["ppr_recall"] = {
                k: sum(r["ppr_recall"][k] for r in rows) / len(rows) for k in ks
            }
        if "mrr" in wanted:
            metrics["ppr_mrr"] = sum(r["ppr_mrr"] for r in rows) / len(rows)
        if "mean_rr" in wanted:
            metrics["ppr_mean_rr"] = sum(r["ppr_mean_rr"] for r in rows) / len(rows)
        if "first_relevant_rank" in wanted:
            ranks = [r.get("ppr_first_relevant_rank") for r in rows]
            present = [rank for rank in ranks if rank is not None]
            metrics["ppr_mean_first_relevant_rank"] = (
                sum(present) / len(present) if present else None
            )
            metrics["ppr_first_relevant_at_k"] = {
                k: sum(
                    (r.get("ppr_first_relevant_rank") or 10**9) <= k for r in rows
                )
                / len(rows)
                for k in ks
            }
    if "extract_embed" in enabled:
        if "recall" in wanted:
            metrics["extract_embed_recall"] = {
                k: sum(r["extract_embed_recall"][k] for r in rows) / len(rows)
                for k in ks
            }
        if "mrr" in wanted:
            metrics["extract_embed_mrr"] = (
                sum(r["extract_embed_mrr"] for r in rows) / len(rows)
            )
        if "mean_rr" in wanted:
            metrics["extract_embed_mean_rr"] = (
                sum(r["extract_embed_mean_rr"] for r in rows) / len(rows)
            )
        if "first_relevant_rank" in wanted:
            ranks = [r.get("extract_embed_first_relevant_rank") for r in rows]
            present = [rank for rank in ranks if rank is not None]
            metrics["extract_embed_mean_first_relevant_rank"] = (
                sum(present) / len(present) if present else None
            )
            metrics["extract_embed_first_relevant_at_k"] = {
                k: sum(
                    (r.get("extract_embed_first_relevant_rank") or 10**9) <= k
                    for r in rows
                )
                / len(rows)
                for k in ks
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
        if "isummary" in enabled:
            print(
                f"iSummary@{primary_k} accuracy:          {metrics['isummary_accuracy']:.1%}"
            )
        if "ppr" in enabled:
            print(f"PPR@{primary_k} accuracy:               {metrics['ppr_accuracy']:.1%}")
        if "extract_embed" in enabled:
            print(
                f"extract+embed@{primary_k} accuracy:      {metrics['extract_embed_accuracy']:.1%}"
            )
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
    if "gss" in enabled:
        if "mrr" in wanted and "gss_mrr" in metrics:
            print(f"GSS MRR (first relevant):     {metrics['gss_mrr']:.3f}")
        if "mean_rr" in wanted and "gss_mean_rr" in metrics:
            print(f"GSS Mean RR (all gold):       {metrics['gss_mean_rr']:.3f}")
        if "first_relevant_rank" in wanted:
            mean_rank = metrics.get("gss_mean_first_relevant_rank")
            if mean_rank is not None:
                print(f"GSS mean first relevant rank: {mean_rank:.3f}")
        if "recall" in wanted or "accuracy" in wanted:
            print("GSS Recall@k / accuracy@k:")
            for k in metrics["ks"]:
                parts = [f"k={k:<3}"]
                if "recall" in wanted and "gss_recall" in metrics:
                    parts.append(f"recall={metrics['gss_recall'][k]:.1%}")
                if "accuracy" in wanted:
                    parts.append(f"gss_acc={metrics['gss_accuracy_at_k'][k]:.1%}")
                print("  " + "  ".join(parts))
    if "isummary" in enabled:
        if "mrr" in wanted and "isummary_mrr" in metrics:
            print(f"iSummary MRR (first relevant): {metrics['isummary_mrr']:.3f}")
        if "mean_rr" in wanted and "isummary_mean_rr" in metrics:
            print(f"iSummary Mean RR (all gold):   {metrics['isummary_mean_rr']:.3f}")
        if "first_relevant_rank" in wanted:
            mean_rank = metrics.get("isummary_mean_first_relevant_rank")
            if mean_rank is not None:
                print(f"iSummary mean first relevant:  {mean_rank:.3f}")
        if "recall" in wanted or "accuracy" in wanted:
            print("iSummary Recall@k / accuracy@k:")
            for k in metrics["ks"]:
                parts = [f"k={k:<3}"]
                if "recall" in wanted and "isummary_recall" in metrics:
                    parts.append(f"recall={metrics['isummary_recall'][k]:.1%}")
                if "accuracy" in wanted:
                    parts.append(
                        f"isummary_acc={metrics['isummary_accuracy_at_k'][k]:.1%}"
                    )
                print("  " + "  ".join(parts))
    if "ppr" in enabled:
        if "mrr" in wanted and "ppr_mrr" in metrics:
            print(f"PPR MRR (first relevant):     {metrics['ppr_mrr']:.3f}")
        if "mean_rr" in wanted and "ppr_mean_rr" in metrics:
            print(f"PPR Mean RR (all gold):       {metrics['ppr_mean_rr']:.3f}")
        if "first_relevant_rank" in wanted:
            mean_rank = metrics.get("ppr_mean_first_relevant_rank")
            if mean_rank is not None:
                print(f"PPR mean first relevant rank: {mean_rank:.3f}")
        if "recall" in wanted or "accuracy" in wanted:
            print("PPR Recall@k / accuracy@k:")
            for k in metrics["ks"]:
                parts = [f"k={k:<3}"]
                if "recall" in wanted and "ppr_recall" in metrics:
                    parts.append(f"recall={metrics['ppr_recall'][k]:.1%}")
                if "accuracy" in wanted:
                    parts.append(f"ppr_acc={metrics['ppr_accuracy_at_k'][k]:.1%}")
                print("  " + "  ".join(parts))
    if "extract_embed" in enabled:
        if "mrr" in wanted and "extract_embed_mrr" in metrics:
            print(
                f"extract+embed MRR (first relevant): {metrics['extract_embed_mrr']:.3f}"
            )
        if "mean_rr" in wanted and "extract_embed_mean_rr" in metrics:
            print(
                f"extract+embed Mean RR (all gold):   {metrics['extract_embed_mean_rr']:.3f}"
            )
        if "first_relevant_rank" in wanted:
            mean_rank = metrics.get("extract_embed_mean_first_relevant_rank")
            if mean_rank is not None:
                print(f"extract+embed mean first relevant: {mean_rank:.3f}")
        if "recall" in wanted or "accuracy" in wanted:
            print("extract+embed Recall@k / accuracy@k:")
            for k in metrics["ks"]:
                parts = [f"k={k:<3}"]
                if "recall" in wanted and "extract_embed_recall" in metrics:
                    parts.append(f"recall={metrics['extract_embed_recall'][k]:.1%}")
                if "accuracy" in wanted:
                    parts.append(
                        f"extract_embed_acc={metrics['extract_embed_accuracy_at_k'][k]:.1%}"
                    )
                print("  " + "  ".join(parts))
    if "failures" in wanted:
        print("Failure tags:", metrics.get("failure_counts", {}))
