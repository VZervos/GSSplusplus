"""Run selected CoLoTa answering approaches from a config file.

Usage (from repo root):

    python src/colota_demo/run_suite.py
    python src/colota_demo/run_suite.py --config src/colota_demo/config.json
    python src/colota_demo/run_suite.py --experiment my_run --data benchmarks/CoLoTa-main/CoLoTa_qa.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from approaches import get_approach, known_approaches
from utils.dataset import (
    DEMO_IDS,
    ROOT,
    build_corpus,
    gold_label,
    item_triples,
    load_qa,
    select_items,
)
from utils.gss_ner import resolve_cache_dir, resolve_seed_source, seed_names
from utils.llm_client import OLLAMA_MODEL
from utils.metrics import (
    ORIGINAL_METRICS,
    classify_failures,
    print_report,
    summarize,
    write_results,
)

DEFAULT_CONFIG = HERE / "config.json"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_ids(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_approaches(raw: str | None) -> dict[str, bool] | None:
    if not raw:
        return None
    selected = {part.strip() for part in raw.split(",") if part.strip()}
    unknown = sorted(selected - set(known_approaches()))
    if unknown:
        raise SystemExit(
            f"Unknown approaches: {unknown}. Known: {known_approaches()}"
        )
    return {name: name in selected for name in known_approaches()}


def selected_approaches(cfg: dict) -> list[str]:
    raw = cfg.get("approaches")
    known = known_approaches()
    if isinstance(raw, dict):
        unknown = [name for name in raw if name not in known]
        if unknown:
            raise SystemExit(
                f"Unknown approaches in config: {unknown}. Known: {known}"
            )
        names = [name for name in known if raw.get(name)]
    elif isinstance(raw, list):
        unknown = [name for name in raw if name not in known]
        if unknown:
            raise SystemExit(
                f"Unknown approaches in config: {unknown}. Known: {known}"
            )
        names = [name for name in known if name in raw]
    else:
        names = []
    if not names:
        raise SystemExit("No approaches selected. Set config.approaches to true.")
    return names


def selected_metrics(cfg: dict) -> set[str]:
    raw = cfg.get("metrics") or {name: True for name in ORIGINAL_METRICS}
    if isinstance(raw, list):
        chosen = {name for name in raw if name}
    else:
        chosen = {name for name, on in raw.items() if on}
    unknown = sorted(chosen - set(ORIGINAL_METRICS))
    if unknown:
        raise SystemExit(
            f"Unknown metrics: {unknown}. Supported: {list(ORIGINAL_METRICS)}"
        )
    return chosen


def safe_experiment_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", name.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip(" ._")
    return cleaned or "experiment"


def default_experiment_name(enabled: list[str]) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    approaches = "-".join(enabled) or "none"
    return f"{stamp}_{approaches}"


def resolve_data_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--ids", type=str, default=None)
    parser.add_argument("--approaches", type=str, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--ks", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--embed-model", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Entity seed source for ppr/extract_embed: gold or gss_ner",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    if args.experiment is not None:
        cfg["experiment"] = args.experiment
    if args.data is not None:
        cfg["data"] = str(args.data)
    if args.ids is not None:
        cfg["ids"] = parse_ids(args.ids)
    if args.approaches is not None:
        cfg["approaches"] = parse_approaches(args.approaches)
    if args.k is not None:
        cfg["primary_k"] = args.k
    if args.ks is not None:
        cfg["ks"] = [int(x) for x in args.ks.split(",") if x.strip()]
    if args.model is not None:
        cfg["model"] = args.model
    if args.embed_model is not None:
        cfg["embed_model"] = args.embed_model
    if args.out is not None:
        cfg["out_dir"] = str(args.out)
    if args.seeds is not None:
        cfg["entity_seeds"] = args.seeds

    cfg.setdefault("model", OLLAMA_MODEL)
    cfg.setdefault("primary_k", 5)
    cfg.setdefault("ks", [1, 3, 5, 10])
    cfg.setdefault("out_dir", "out/colota_demo")
    if not isinstance(cfg.get("gss"), dict):
        cfg["gss"] = {"a": 0.75, "kg": "corpus"}
    else:
        cfg["gss"].setdefault("a", 0.75)
        cfg["gss"].setdefault("kg", "corpus")
    cfg.setdefault("entity_seeds", "gold")
    cfg.setdefault("ner_cache", "out/colota_demo/gss_ner_cache")
    if not isinstance(cfg.get("ppr"), dict):
        cfg["ppr"] = {"alpha": 0.85}
    else:
        cfg["ppr"].setdefault("alpha", 0.85)
    if not isinstance(cfg.get("isummary"), dict):
        cfg["isummary"] = {
            "kappa": 12,
            "jar": "third_party/iSummary/isummary.jar",
            "train": "third_party/iSummary/data/wikidata/train.txt",
            "test": "src/colota_demo/approaches/isummary/java/dummy_test.tsv",
            "timeout_sec": 900,
        }
    else:
        cfg["isummary"].setdefault("kappa", 12)
        cfg["isummary"].setdefault("jar", "third_party/iSummary/isummary.jar")
        cfg["isummary"].setdefault(
            "train", "third_party/iSummary/data/wikidata/train.txt"
        )
        cfg["isummary"].setdefault(
            "test", "src/colota_demo/approaches/isummary/java/dummy_test.tsv"
        )
        cfg["isummary"].setdefault("timeout_sec", 900)
    return cfg


def main() -> None:
    cfg = resolve_config(build_parser().parse_args())
    enabled = selected_approaches(cfg)
    enabled_set = set(enabled)
    wanted = selected_metrics(cfg)
    ks = list(cfg["ks"])
    primary_k = cfg["primary_k"]
    if (
        "rag_transformers" in enabled_set
        or "gss" in enabled_set
        or "isummary" in enabled_set
        or "ppr" in enabled_set
        or "extract_embed" in enabled_set
    ) and primary_k not in ks:
        ks = sorted(set(ks) | {primary_k})

    data_path = resolve_data_path(cfg.get("data"))
    ids = cfg.get("ids") or DEMO_IDS
    data = load_qa(data_path)
    items = select_items(data, ids)
    corpus = build_corpus(items)
    model = cfg["model"]

    experiment = cfg.get("experiment") or default_experiment_name(enabled)
    experiment = safe_experiment_name(str(experiment))
    out_base = Path(cfg["out_dir"])
    if not out_base.is_absolute():
        out_base = ROOT / out_base
    out_dir = out_base / experiment

    runners = {}
    for name in enabled:
        approach = get_approach(name)
        approach.prepare(corpus, {**cfg, "out_dir": str(out_dir)})
        runners[name] = approach

    print(
        f"Suite: experiment={experiment} n={len(items)} corpus={len(corpus)} "
        f"model={model} approaches={enabled}"
    )

    rows = []
    for item in items:
        question = item["query"]
        gold = gold_label(item)
        triples = item_triples(item)
        row: dict = {
            "id": item["id"],
            "question": question,
            "gold": gold,
            "n_gold_triples": len(triples),
        }

        if "llm_only" in runners:
            row["llm"] = runners["llm_only"].predict(question, model=model)
        if "gold_triples" in runners:
            row["gold_triples"] = runners["gold_triples"].predict(
                question, item=item, model=model
            )
        if "rag_transformers" in runners:
            rag = runners["rag_transformers"]
            ranking = rag.rank(question)
            ranks = rag.evaluate_ranking(triples, ranking, ks)
            row["mrr"] = ranks["mrr"]
            row["mean_rr"] = ranks["mean_rr"]
            row["first_relevant_rank"] = ranks["first_relevant_rank"]
            row["gold_ranks"] = ranks["gold_ranks"]
            row["recall"] = ranks["recall"]
            row["rag"] = {
                k: rag.predict(question, model=model, k=k, ranking=ranking)
                for k in ks
            }
        if "gss" in runners:
            gss = runners["gss"]
            gss_ranking = gss.summarize(question, item_id=item["id"])
            gss_ranks = gss.evaluate_ranking(triples, gss_ranking, ks)
            row["gss_mrr"] = gss_ranks["mrr"]
            row["gss_mean_rr"] = gss_ranks["mean_rr"]
            row["gss_first_relevant_rank"] = gss_ranks["first_relevant_rank"]
            row["gss_gold_ranks"] = gss_ranks["gold_ranks"]
            row["gss_recall"] = gss_ranks["recall"]
            row["gss"] = {
                k: gss.predict(
                    question,
                    model=model,
                    k=k,
                    ranking=gss_ranking,
                    item_id=item["id"],
                )
                for k in ks
            }

        if "isummary" in runners:
            isum = runners["isummary"]
            isum_ranking = isum.summarize(
                question, item=item, item_id=item["id"]
            )
            isum_ranks = isum.evaluate_ranking(triples, isum_ranking, ks)
            row["isummary_mrr"] = isum_ranks["mrr"]
            row["isummary_mean_rr"] = isum_ranks["mean_rr"]
            row["isummary_first_relevant_rank"] = isum_ranks["first_relevant_rank"]
            row["isummary_gold_ranks"] = isum_ranks["gold_ranks"]
            row["isummary_recall"] = isum_ranks["recall"]
            row["isummary"] = {
                k: isum.predict(
                    question,
                    model=model,
                    k=k,
                    ranking=isum_ranking,
                    item=item,
                    item_id=item["id"],
                )
                for k in ks
            }

        if "ppr" in runners or "extract_embed" in runners:
            row["entity_seeds"] = resolve_seed_source(cfg)
            row["seed_names"] = seed_names(item, question, cfg)
            row["n_seed_names"] = len(row["seed_names"])
            row["ner_cache"] = str(resolve_cache_dir(cfg))

        if "ppr" in runners:
            ppr = runners["ppr"]
            ppr_ranking = ppr.rank(question, item=item)
            ppr_ranks = ppr.evaluate_ranking(triples, ppr_ranking, ks)
            row["ppr_mrr"] = ppr_ranks["mrr"]
            row["ppr_mean_rr"] = ppr_ranks["mean_rr"]
            row["ppr_first_relevant_rank"] = ppr_ranks["first_relevant_rank"]
            row["ppr_gold_ranks"] = ppr_ranks["gold_ranks"]
            row["ppr_recall"] = ppr_ranks["recall"]
            row["ppr"] = {
                k: ppr.predict(
                    question,
                    model=model,
                    k=k,
                    ranking=ppr_ranking,
                    item=item,
                )
                for k in ks
            }

        if "extract_embed" in runners:
            ext = runners["extract_embed"]
            ext_ranking = ext.rank(question, item=item)
            ext_ranks = ext.evaluate_ranking(triples, ext_ranking, ks)
            row["extract_embed_mrr"] = ext_ranks["mrr"]
            row["extract_embed_mean_rr"] = ext_ranks["mean_rr"]
            row["extract_embed_first_relevant_rank"] = ext_ranks[
                "first_relevant_rank"
            ]
            row["extract_embed_gold_ranks"] = ext_ranks["gold_ranks"]
            row["extract_embed_recall"] = ext_ranks["recall"]
            row["extract_embed"] = {
                k: ext.predict(
                    question,
                    model=model,
                    k=k,
                    ranking=ext_ranking,
                    item=item,
                )
                for k in ks
            }

        if "failures" in wanted:
            row["failures"] = classify_failures(row, primary_k, enabled_set)
        rows.append(row)
        bits = [item["id"], f"gold={gold}"]
        if "llm" in row:
            bits.append(f"llm={row['llm']}")
        if "gold_triples" in row:
            bits.append(f"gold_triples={row['gold_triples']}")
        if "rag" in row:
            bits.append(f"rag@{primary_k}={row['rag'][primary_k]}")
            if "recall" in row:
                bits.append(f"R@{primary_k}={row['recall'][primary_k]:.0%}")
        if "gss" in row:
            bits.append(f"gss@{primary_k}={row['gss'][primary_k]}")
            if "gss_recall" in row:
                bits.append(f"gssR@{primary_k}={row['gss_recall'][primary_k]:.0%}")
        if "isummary" in row:
            bits.append(f"isummary@{primary_k}={row['isummary'][primary_k]}")
            if "isummary_recall" in row:
                bits.append(
                    f"isummaryR@{primary_k}={row['isummary_recall'][primary_k]:.0%}"
                )
        if "seed_names" in row:
            bits.append(f"seeds={row['n_seed_names']}")
        if "ppr" in row:
            bits.append(f"ppr@{primary_k}={row['ppr'][primary_k]}")
            if "ppr_recall" in row:
                bits.append(f"pprR@{primary_k}={row['ppr_recall'][primary_k]:.0%}")
        if "extract_embed" in row:
            bits.append(
                f"extract_embed@{primary_k}={row['extract_embed'][primary_k]}"
            )
            if "extract_embed_recall" in row:
                bits.append(
                    f"extract_embedR@{primary_k}={row['extract_embed_recall'][primary_k]:.0%}"
                )
        print("  " + "  ".join(bits))

    metrics = summarize(rows, enabled_set, ks, primary_k, wanted)
    metrics["corpus_size"] = len(corpus)
    metrics["model"] = model
    metrics["experiment"] = experiment
    metrics["data"] = str(data_path) if data_path else None
    metrics["config"] = {
        "approaches": enabled,
        "ks": ks,
        "primary_k": primary_k,
        "embed_model": cfg.get("embed_model"),
        "gss": cfg.get("gss"),
        "ppr": cfg.get("ppr"),
        "entity_seeds": cfg.get("entity_seeds"),
        "ner_cache": cfg.get("ner_cache"),
        "isummary": cfg.get("isummary"),
    }
    write_results(out_dir, rows, metrics, enabled_set, wanted)
    (out_dir / "config.json").write_text(
        json.dumps(cfg, indent=2),
        encoding="utf-8",
    )
    print_report(metrics, enabled_set, primary_k, wanted)
    print(f"Wrote {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
