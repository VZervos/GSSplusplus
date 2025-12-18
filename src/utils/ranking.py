from collections import defaultdict

def compute_final_scores(triples, importance_weight=0.6, similarity_weight=0.4) -> None:
    """
    Combine graph-based importance and semantic similarity into a single final score.
    
    Both importance and similarity are in [0, 1] range, so the final score
    is also in [0, 1] range: IMPORTANCE_WEIGHT * importance + SIMILARITY_WEIGHT * similarity
    
    Args:
        triples: List of triples to score
        importance_weight: Weight for importance score (default: 0.6)
        similarity_weight: Weight for similarity score (default: 0.4)
    """
    for t in triples:
        importance = t.get("importance", 0.0)
        similarity = t.get("similarity", 0.0)

        # Both importance and similarity are in [0, 1], so final score is in [0, 1]
        t["final_score"] = (
            importance_weight * importance +
            similarity_weight * similarity
        )

def select_subgraph_triples(
    triples: list,
    uri_importance_map: dict,
    per_entity_limit: int = 45, # How many triples is each entity allowed to contribute?
    min_score: float = 0.1, # Minimum score to include a triple
    max_total: int = 350 # Maximum total number of triples to include
) -> list:
    """
    Build a query-focused subgraph instead of a Top-K answer list.

    Strategy:
    - Group triples by involved entities
    - For each important entity, keep its best triples
    - Apply light pruning to remove pure noise
    """

    triples_by_entity = defaultdict(list)

    # Group triples by subject and object
    for t in triples:
        if t.get("s"):
            triples_by_entity[t["s"]].append(t)
        if t.get("o"):
            triples_by_entity[t["o"]].append(t)

    selected = []
    seen = set()

    # Process entities in order of importance
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

    # cap the subgraph size
    if len(selected) > max_total:
        selected = sorted(
            selected,
            key=lambda t: t.get("final_score", 0.0),
            reverse=True
        )[:max_total]

    return selected