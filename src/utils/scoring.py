from utils.dbpedia import (
    fetch_triples_batch,
    extract_entity_name_from_uri,
    is_predicate_uri,
    is_resource_uri,
    compute_weighted_degree
)


def _normalize_triple_importance(score_sum: float) -> float:
    """Normalizes triple importance score sum to [0, 1] range.
    
    Args:
        score_sum: Sum of entity importance scores (subject + object + predicate)
                   Each entity score is in [0, 1], so max sum is 3.0
    
    Returns:
        Normalized importance score in [0, 1] range
    """
    # Maximum possible sum is 3.0 (if all three entities have max importance of 1.0)
    max_sum = 3.0
    return min(1.0, score_sum / max_sum) if max_sum > 0 else 0.0


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
    # If we have many resources/predicates, split into smaller batches to avoid query size issues
    print(f"  Fetching triples batch: {len(resource_uris)} resources, {len(predicate_uris)} predicates")
    
    batch_triples = []
    if len(resource_uris) > 10 or (len(resource_uris) > 5 and len(predicate_uris) > 3):
        # Split into smaller batches to avoid query size issues
        print(f"    Splitting into smaller batches to avoid query size limits...")
        # Process resources in smaller chunks
        resource_batch_size = 5
        predicate_batch_size = 3
        for i in range(0, len(resource_uris), resource_batch_size):
            batch_resources = resource_uris[i:i+resource_batch_size]
            for j in range(0, len(predicate_uris), predicate_batch_size):
                batch_predicates = predicate_uris[j:j+predicate_batch_size]
                limit = triples_per_entity_limit * max(len(batch_resources), len(batch_predicates), 1)
                sub_batch = fetch_triples_batch(batch_resources, batch_predicates, limit=limit)
                batch_triples.extend(sub_batch)
                print(f"      Batch {i//resource_batch_size + 1}-{j//predicate_batch_size + 1}: {len(sub_batch)} triples")
    else:
        # Small enough to query in one go
        limit = triples_per_entity_limit * max(len(resource_uris), len(predicate_uris), 1)
        batch_triples = fetch_triples_batch(resource_uris, predicate_uris, limit=limit)
        print(f"    Retrieved {len(batch_triples)} triples from batch query")
    
    all_triples.extend(batch_triples)
    
    # Count predicate usage with our resources (for reporting only)
    predicate_triple_counts = {}
    if predicate_uris and resource_uris:
        resource_set = set(resource_uris)
        predicate_set = set(predicate_uris)
        for triple in batch_triples:
            predicate_uri = triple.get("p", "")
            if predicate_uri in predicate_set:
                subject = triple.get("s", "")
                object_uri = triple.get("o", "")
                if subject in resource_set or object_uri in resource_set:
                    predicate_triple_counts[predicate_uri] = predicate_triple_counts.get(predicate_uri, 0) + 1
    
    # Compute importance for each URI using only initial importance (1-5)
    # Use power function: importance^2, then normalize to [0, 1] range
    # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=25 -> normalized: (value/25)
    # Map: 1 -> 0.04, 2 -> 0.16, 3 -> 0.36, 4 -> 0.64, 5 -> 1.0, 0 (not found) -> 0.0
    for entity, uri in entity_uris.items():
        print(f"  Computing importance for: {entity}")
        
        importance_weight = entity_importance.get(uri, 0)  # 0 if not found
        
        if importance_weight == 0:
            # URI doesn't exist or wasn't in initial keywords
            score = 0.0
            out_degree = in_degree = total_degree = 0
        else:
            # Use power function: importance^2, then normalize to [0, 1]
            # Max value is 5^2 = 25, so normalize by (value / 25)
            squared_value = importance_weight ** 2
            score = squared_value / 25.0
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
    """Assigns importance scores to triples based on the sum of entity scores, normalized to [0, 1]."""
    for triple in all_triples:
        # Sum scores of subject, object, and predicate
        score_sum = 0.0
        for key in ["s", "o", "p"]:
            uri = triple.get(key, "")
            if uri.startswith("http://dbpedia.org/") and uri in uri_importance_map:
                score_sum += uri_importance_map[uri]["score"]
        
        # Normalize to [0, 1] range (max sum is 3.0 if all entities have max importance)
        triple["importance"] = _normalize_triple_importance(score_sum)
        triple["_s_name"] = extract_entity_name_from_uri(triple.get("s", ""))
        triple["_o_name"] = extract_entity_name_from_uri(triple.get("o", ""))
        triple["_query_keywords"] = query_keywords
