from collections import defaultdict

from config.settings import (
    SCORING_IMPORTANCE_WEIGHT,
    SCORING_SIMILARITY_WEIGHT,
    RANKING_PER_ENTITY_LIMIT,
    RANKING_MIN_SCORE,
    RANKING_MAX_TOTAL
)


def compute_final_scores(triples, importance_weight=None, similarity_weight=None) -> None:
    """
    Combine graph-based importance and semantic similarity into a single final score.
    
    Both importance and similarity are in [0, 1] range, so the final score
    is also in [0, 1] range: IMPORTANCE_WEIGHT * importance + SIMILARITY_WEIGHT * similarity
    
    Args:
        triples: List of triples to score
        importance_weight: Weight for importance score (default: from settings)
        similarity_weight: Weight for similarity score (default: from settings)
    """
    if importance_weight is None:
        importance_weight = SCORING_IMPORTANCE_WEIGHT
    if similarity_weight is None:
        similarity_weight = SCORING_SIMILARITY_WEIGHT

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
        per_entity_limit: int = None,
        min_score: float = None,
        max_total: int = None
) -> list:
    """
    Build a query-focused subgraph instead of a Top-K answer list.

    Strategy:
    - Group triples by involved entities
    - For each important entity, keep its best triples
    - Apply light pruning to remove pure noise
    
    Args:
        triples: List of triples to select from
        uri_importance_map: Dict mapping URIs to importance information
        per_entity_limit: How many triples each entity can contribute (default: from settings)
        min_score: Minimum final score to include a triple (default: from settings)
        max_total: Maximum total number of triples in subgraph (default: from settings)
    """
    if per_entity_limit is None:
        per_entity_limit = RANKING_PER_ENTITY_LIMIT
    if min_score is None:
        min_score = RANKING_MIN_SCORE
    if max_total is None:
        max_total = RANKING_MAX_TOTAL

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
