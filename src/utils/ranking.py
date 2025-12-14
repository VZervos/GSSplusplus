def compute_final_scores(triples, importance_weight=0.6, similarity_weight=0.4) -> None:
    """
    Combine graph-based importance and semantic similarity
    into a single final score.
    """
    for t in triples:
        importance = t.get("importance", 0.0)
        similarity = t.get("similarity", 0.0)

        t["final_score"] = (
            importance_weight * importance +
            similarity_weight * similarity * 10.0
        )

def rank_and_select(triples: list, k: int = 20) -> list:
    """
    Sort triples by final score and return Top-K.
    """
    triples.sort(key=lambda t: t.get("final_score", 0.0), reverse=True)
    return triples[:k]
