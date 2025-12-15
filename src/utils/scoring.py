from utils.dbpedia import (
    fetch_triples_batch,
    extract_entity_name_from_uri,
    is_predicate_uri,
    is_resource_uri,
    compute_weighted_degree
)


def _normalize_importance_score(degree: int) -> float:
    """Normalizes degree to importance score in 0-10 range."""
    return min(10.0, 1.0 + (degree ** 0.5) / 10.0) if degree > 0 else 0.0


def _calculate_importance_multiplier(importance_weight: int) -> float:
    """Calculates multiplier based on importance weight (1, 2, or 3)."""
    return 1.0 + (importance_weight - 1) * 0.5


def compute_importance(all_triples: list, entity_uris: dict, uri_importance_map: dict, entity_importance: dict = None, triples_per_entity_limit: int = 1000) -> None:
    """
    Computes importance scores for entities using only initial importance weights (1-5).
    Does not use in/out degree computation.
    
    Args:
        all_triples: List to store retrieved triples
        entity_uris: Dict mapping entity names to URIs
        uri_importance_map: Dict to store importance scores for URIs
        entity_importance: Dict mapping URIs to importance values (1, 2, 3, 4, or 5)
        triples_per_entity_limit: Limit for triples per entity
    """
    if entity_importance is None:
        entity_importance = {}
    
    # Separate resources and predicates
    resource_uris = []
    predicate_uris = []
    for entity, uri in entity_uris.items():
        if is_predicate_uri(uri):
            predicate_uris.append(uri)
        elif is_resource_uri(uri):
            resource_uris.append(uri)
    
    # Fetch triples in batch
    print(f"  Fetching triples batch: {len(resource_uris)} resources, {len(predicate_uris)} predicates")
    limit = triples_per_entity_limit * max(len(resource_uris), len(predicate_uris), 1)
    batch_triples = fetch_triples_batch(resource_uris, predicate_uris, limit=limit)
    all_triples.extend(batch_triples)
    print(f"    Retrieved {len(batch_triples)} triples from batch query")
    
    # Count predicate usage with our resources (for reporting only)
    predicate_triple_counts = {}
    if predicate_uris and resource_uris:
        resource_set = set(resource_uris)
        for triple in batch_triples:
            predicate_uri = triple.get("p", "")
            if predicate_uri in predicate_uris:
                subject = triple.get("s", "")
                object_uri = triple.get("o", "")
                if subject in resource_set or object_uri in resource_set:
                    predicate_triple_counts[predicate_uri] = predicate_triple_counts.get(predicate_uri, 0) + 1
    
    # Compute importance for each URI using only initial importance (1-5)
    # Use power function: importance^2, then normalize to 0-10 range
    # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=25 -> normalized: (value/25)*10
    # Map: 1 -> 0.4, 2 -> 1.6, 3 -> 3.6, 4 -> 6.4, 5 -> 10.0, 0 (not found) -> 0.0
    for entity, uri in entity_uris.items():
        print(f"  Computing importance for: {entity}")
        
        importance_weight = entity_importance.get(uri, 0)  # 0 if not found
        
        if importance_weight == 0:
            # URI doesn't exist or wasn't in initial keywords
            score = 0.0
            out_degree = in_degree = total_degree = 0
        else:
            # Use power function: importance^2, then normalize to 0-10
            # Max value is 5^2 = 25, so normalize by (value / 25) * 10
            squared_value = importance_weight ** 2
            score = (squared_value / 25.0) * 10.0
            # Still compute degrees for reporting, but don't use them for score
            if is_predicate_uri(uri):
                total_degree = predicate_triple_counts.get(uri, 0)
                out_degree = total_degree
                in_degree = 0
            else:
                out_degree, in_degree, total_degree = compute_weighted_degree(uri)
        
        uri_importance_map[uri] = {
            "method": "initial_importance_only",
            "score": score,
            "base_score": score,
            "importance_weight": importance_weight,
            "out_degree": out_degree,
            "in_degree": in_degree,
            "total_degree": total_degree
        }
        
        print(f"    Importance: {score:.2f} (weight: {importance_weight}, degree: {total_degree})")


def assign_importance_scores(all_triples: list, query_keywords: set, uri_importance_map: dict) -> None:
    """Assigns importance scores to triples based on the sum of entity scores, normalized."""
    for triple in all_triples:
        # Sum scores of subject, object, and predicate
        score_sum = 0.0
        for key in ["s", "o", "p"]:
            uri = triple.get(key, "")
            if uri.startswith("http://dbpedia.org/") and uri in uri_importance_map:
                score_sum += uri_importance_map[uri]["score"]
        
        triple["importance"] = _normalize_importance_score(score_sum)
        triple["_s_name"] = extract_entity_name_from_uri(triple.get("s", ""))
        triple["_o_name"] = extract_entity_name_from_uri(triple.get("o", ""))
        triple["_query_keywords"] = query_keywords
