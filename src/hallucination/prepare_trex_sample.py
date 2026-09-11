"""Build a small FactEval-style T-REx question sample from LAMA data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TEMPLATES_WH = {
    "P19": "Where was {sub} born?",
    "P20": "Where did {sub} die?",
    "P27": "What is the country of citizenship of {sub}?",
    "P36": "What is the capital of {sub}?",
    "P106": "What is the occupation of {sub}?",
    "P17": "Which country is {sub} located in?",
    "P159": "Where is the headquarters of {sub}?",
    "P495": "In which country was {sub} created?",
    "P740": "Where was {sub} formed?",
    "P937": "Where does {sub} work?",
}

PREFERRED = list(TEMPLATES_WH.keys())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _row_to_sample(row: dict, rel: str, relations: dict, idx: int) -> dict:
    info = relations[rel]
    question = TEMPLATES_WH[rel].format(sub=row["sub_label"])
    return {
        "id": f"trex_{rel}_{row.get('uuid', idx)}",
        "question": question,
        "gold_answers": [{"label": row["obj_label"], "uri": row["obj_uri"]}],
        "subject": {"label": row["sub_label"], "uri": row["sub_uri"]},
        "relation": info,
        "source": "LAMA-TREx",
        "triple": {
            "sub_label": row["sub_label"],
            "obj_label": row["obj_label"],
            "sub_uri": row["sub_uri"],
            "obj_uri": row["obj_uri"],
            "predicate_id": rel,
        },
    }


def build_sample(n: int = 10, seed: int = 42) -> dict:
    """Build a stratified T-REx sample.

    Draws round-robin across PREFERRED relations so n>10 still covers all
    relations (e.g. n=30 → about 3 items per relation).
    """
    random.seed(seed)
    root = _repo_root() / "dataset" / "facteval_trex" / "lama" / "data"
    relations = {}
    with (root / "relations.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rel = json.loads(line)
            relations[rel["relation"]] = rel

    pools: dict[str, list[dict]] = {}
    for rel in PREFERRED:
        path = root / "TREx" / f"{rel}.jsonl"
        if not path.exists():
            continue
        rows = []
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 800:
                    break
                row = json.loads(line)
                obj = (row.get("obj_label") or "").strip()
                sub = (row.get("sub_label") or "").strip()
                if sub and obj and len(obj.split()) <= 4:
                    rows.append(row)
        if rows:
            random.shuffle(rows)
            pools[rel] = rows

    samples = []
    seen_ids: set[str] = set()
    # Round-robin across relations until we hit n
    while len(samples) < n and pools:
        progressed = False
        for rel in list(PREFERRED):
            if len(samples) >= n:
                break
            rows = pools.get(rel)
            if not rows:
                pools.pop(rel, None)
                continue
            row = rows.pop(0)
            sample = _row_to_sample(row, rel, relations, len(samples))
            if sample["id"] in seen_ids:
                continue
            seen_ids.add(sample["id"])
            samples.append(sample)
            progressed = True
            if not rows:
                pools.pop(rel, None)
        if not progressed:
            break

    return {"benchmark": "FactEval-TREx", "n": len(samples), "seed": seed, "questions": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small T-REx FactEval sample")
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(_repo_root() / "dataset" / "facteval_trex" / "sample_10.json"),
    )
    args = parser.parse_args()

    sample = build_sample(n=args.n, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {sample['n']} questions to {out}")
    for q in sample["questions"]:
        print(f"  {q['id']}: {q['question']} -> {q['gold_answers'][0]['label']}")


if __name__ == "__main__":
    main()
