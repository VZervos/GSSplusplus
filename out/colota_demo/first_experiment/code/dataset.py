"""Load the hand-picked CoLoTa QA subset and build a pooled triple corpus."""

from __future__ import annotations

import json
from pathlib import Path

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "benchmarks" / "CoLoTa-main" / "CoLoTa_qa.json").exists():
            return parent
    raise FileNotFoundError("Could not find the GSSplusplus repo root")


ROOT = _repo_root()
DEFAULT_QA_PATH = ROOT / "benchmarks" / "CoLoTa-main" / "CoLoTa_qa.json"

# Hand-picked CoLoTa QA subset: entity-aligned gold triples, mixed
# True/False, and distinct reasoning types (geo, occupation, temporal, …).
DEMO_IDS = [
    "S1",    # True  — number comparison (Horsens vs Ikast)
    "S2",    # False — set inclusion (pet breed)
    "S3",    # True  — occupation (singer)
    "S5",    # True  — political / predecessor gender
    "S13",   # False — geographical (taxi continents)
    "S37",   # False — geographical / physical (car travel)
    "S51",   # True  — temporal / sports (Serie A debut)
    "S87",   # False — biological (middle child)
    "S95",   # False — technological (car cannot vlog)
    "S131",  # False — occupation (CEO vs janitor)
    "S142",  # True  — language / geography
    "S160",  # True  — geography / citizenship
    "S164",  # True  — education (math professor)
    "S175",  # False — music / set inclusion
    "S184",  # False — historical / medical (decapitation)
    "S4",    # False — temporal / technology (telephone)
    "S10",   # False — cultural (film rating)
    "S34",   # True  — literature / set inclusion
    "S55",   # True  — medical (melancholia)
    "S124",  # True  — entity comparison (cause of death)
]


def load_qa(path: Path | None = None) -> list[dict]:
    path = path or DEFAULT_QA_PATH
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def select_items(data: list[dict], ids: list[str] | None = None) -> list[dict]:
    wanted = ids or DEMO_IDS
    by_id = {item["id"]: item for item in data}
    missing = [qid for qid in wanted if qid not in by_id]
    if missing:
        raise KeyError(f"CoLoTa ids not found: {missing}")
    return [by_id[qid] for qid in wanted]


def gold_label(item: dict) -> str:
    return "True" if item["answer"] else "False"


def normalize_triple(triple) -> str:
    return str(triple).strip()


def item_triples(item: dict) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for triple in item.get("kg_triples") or []:
        text = normalize_triple(triple)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def build_corpus(items: list[dict]) -> list[dict]:
    """One document per unique triple, tagged with source question ids."""
    by_triple: dict[str, dict] = {}
    for item in items:
        qid = item["id"]
        for triple in item_triples(item):
            if triple not in by_triple:
                by_triple[triple] = {
                    "triple": triple,
                    "text": triple,
                    "source_ids": [qid],
                }
            elif qid not in by_triple[triple]["source_ids"]:
                by_triple[triple]["source_ids"].append(qid)
    return list(by_triple.values())
