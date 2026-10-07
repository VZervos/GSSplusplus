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

    cfg.setdefault("model", OLLAMA_MODEL)
    cfg.setdefault("primary_k", 5)
    cfg.setdefault("ks", [1, 3, 5, 10])
    cfg.setdefault("out_dir", "out/colota_demo")
    if not isinstance(cfg.get("gss"), dict):
        cfg["gss"] = {"a": 0.75}
    else:
        cfg["gss"].setdefault("a", 0.75)
    return cfg


def main() -> None:
    cfg = resolve_config(build_parser().parse_args())
    enabled = selected_approaches(cfg)
    enabled_set = set(enabled)
    wanted = selected_metrics(cfg)
    ks = list(cfg["ks"])
    primary_k = cfg["primary_k"]
    if (
        "rag_transformers" in enabled_set or "gss" in enabled_set
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
