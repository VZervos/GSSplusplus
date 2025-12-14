from utils.dbpedia import (
    fetch_triples_batch,
    extract_entity_name_from_uri,
    is_predicate_uri,
    is_resource_uri,
    compute_weighted_degree
)


def _normalize_importance_score(degree):
    """Normalizes degree to importance score in 0-10 range."""
    return min(10.0, 1.0 + (degree ** 0.5) / 10.0) if degree > 0 else 0.0


def _calculate_importance_multiplier(importance_weight):
    """Calculates multiplier based on importance weight (1, 2, or 3)."""
    return 1.0 + (importance_weight - 1) * 0.5


def compute_importance(all_triples, entity_uris, uri_importance_map, entity_importance=None, triples_per_entity_limit=200):
    """
    Computes importance scores for entities with optional importance weighting.
    
    Args:
        all_triples: List to store retrieved triples
        entity_uris: Dict mapping entity names to URIs
        uri_importance_map: Dict to store importance scores for URIs
        entity_importance: Dict mapping URIs to importance values (1, 2, or 3)
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
    
    # Count predicate usage with our resources
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
    
    # Compute importance for each URI
    for entity, uri in entity_uris.items():
        print(f"  Computing importance for: {entity}")
        
        importance_weight = entity_importance.get(uri, 1)
        
        if is_predicate_uri(uri):
            total_degree = predicate_triple_counts.get(uri, 0)
            out_degree = total_degree
            in_degree = 0
        else:
            out_degree, in_degree, total_degree = compute_weighted_degree(uri)
        
        normalized_score = _normalize_importance_score(total_degree)
        importance_multiplier = _calculate_importance_multiplier(importance_weight)
        weighted_score = min(10.0, normalized_score * importance_multiplier)
        
        uri_importance_map[uri] = {
            "method": "weighted_degree",
            "score": weighted_score,
            "base_score": normalized_score,
            "importance_weight": importance_weight,
            "out_degree": out_degree,
            "in_degree": in_degree,
            "total_degree": total_degree
        }
        
        print(f"    Importance: {weighted_score:.2f} (base: {normalized_score:.2f}, weight: {importance_weight}x{importance_multiplier:.1f}, degree: {total_degree})")


def assign_importance_scores(all_triples, query_keywords, uri_importance_map):
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
