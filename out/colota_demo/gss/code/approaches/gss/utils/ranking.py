from collections import defaultdict


def compute_final_scores(triples, a=None) -> None:
    """Combine graph-based importance and semantic similarity: a * I + (1-a) * S."""
    if a is None:
        from ..config.settings import A
        a = A
    a = float(a)
    for t in triples:
        importance = t.get("importance", 0.0)
        similarity = t.get("similarity", 0.0)
        t["final_score"] = a * importance + (1.0 - a) * similarity


def select_subgraph_triples(
        triples: list,
        uri_importance_map: dict,
        per_entity_limit: int = 45,
        min_score: float = 0.1,
        max_total: int = 350,
) -> list:
    """Build a query-focused subgraph from triples."""

    triples_by_entity = defaultdict(list)

    for t in triples:
        if t.get("s"):
            triples_by_entity[t["s"]].append(t)
        if t.get("o"):
            triples_by_entity[t["o"]].append(t)

    selected = []
    seen = set()

    for uri, _info in sorted(
            uri_importance_map.items(),
            key=lambda x: x[1].get("score", 0.0),
            reverse=True
    ):
        candidates = sorted(
            triples_by_entity.get(uri, []),
            key=lambda t: t.get("final_score", 0.0),
            reverse=True
        )

        kept = 0
        for t in candidates:
            if t.get("final_score", 0.0) < min_score:
                break

            tid = id(t)
            if tid in seen:
                continue

            selected.append(t)
            seen.add(tid)
            kept += 1

            if kept >= per_entity_limit:
                break

    if len(selected) > max_total:
        selected = sorted(
            selected,
            key=lambda t: t.get("final_score", 0.0),
            reverse=True
        )[:max_total]

    return selected
