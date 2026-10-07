SCORING_TRIPLES_PER_ENTITY_LIMIT = 1000
SCORING_IMPORTANCE_POWER = 2
SCORING_IMPORTANCE_MAX_VALUE = 25
SCORING_TRIPLE_IMPORTANCE_MAX_SUM = 3.0
SPLIT_RESOURCE_THRESHOLD = 10
SPLIT_RESOURCE_SMALL_THRESHOLD = 5
SPLIT_PREDICATE_THRESHOLD = 3
BATCH_RESOURCE_SIZE = 5
BATCH_PREDICATE_SIZE = 3
from .dbpedia import (
    fetch_triples_batch,
    extract_entity_name_from_uri,
    is_predicate_uri,
    is_resource_uri,
    compute_weighted_degree
)


def _normalize_triple_importance(score_sum: float) -> float:
    """Normalizes triple importance score sum to [0, 1] range."""
    max_sum = SCORING_TRIPLE_IMPORTANCE_MAX_SUM
    return min(1.0, score_sum / max_sum) if max_sum > 0 else 0.0


def compute_importance(all_triples: list, entity_uris: dict, uri_importance_map: dict, entity_importance: dict = None,
                       triples_per_entity_limit: int = None) -> None:
    """Computes importance scores for entities using only initial importance weights (1-5)."""
    if triples_per_entity_limit is None:
        triples_per_entity_limit = SCORING_TRIPLES_PER_ENTITY_LIMIT

    if entity_importance is None:
        entity_importance = {}

    resource_uris = []
    predicate_uris = []
    for entity, uri in entity_uris.items():
        if is_predicate_uri(uri):
            predicate_uris.append(uri)
        elif is_resource_uri(uri):
            resource_uris.append(uri)

    print(f"  Fetching triples batch: {len(resource_uris)} resources, {len(predicate_uris)} predicates")

    batch_triples = []
    if (len(resource_uris) > SPLIT_RESOURCE_THRESHOLD or
            (len(resource_uris) > SPLIT_RESOURCE_SMALL_THRESHOLD and len(
                predicate_uris) > SPLIT_PREDICATE_THRESHOLD)):
        print(f"    Splitting into smaller batches...")
        resource_batch_size = BATCH_RESOURCE_SIZE
        predicate_batch_size = BATCH_PREDICATE_SIZE
        for i in range(0, len(resource_uris), resource_batch_size):
            batch_resources = resource_uris[i:i + resource_batch_size]
            for j in range(0, len(predicate_uris), predicate_batch_size):
                batch_predicates = predicate_uris[j:j + predicate_batch_size]
                limit = triples_per_entity_limit * max(len(batch_resources), len(batch_predicates), 1)
                sub_batch = fetch_triples_batch(batch_resources, batch_predicates, limit=limit)
                batch_triples.extend(sub_batch)
                print(
                    f"      Batch {i // resource_batch_size + 1}-{j // predicate_batch_size + 1}: {len(sub_batch)} triples")
    else:
        limit = triples_per_entity_limit * max(len(resource_uris), len(predicate_uris), 1)
        batch_triples = fetch_triples_batch(resource_uris, predicate_uris, limit=limit)
        print(f"    Retrieved {len(batch_triples)} triples from batch query")

    all_triples.extend(batch_triples)

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

    for entity, uri in entity_uris.items():
        print(f"  Computing importance for: {entity}")

        importance_weight = entity_importance.get(uri, 0)

        if importance_weight == 0:
            score = 0.0
            out_degree = in_degree = total_degree = 0
        else:
            powered_value = importance_weight ** SCORING_IMPORTANCE_POWER
            score = powered_value / SCORING_IMPORTANCE_MAX_VALUE
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
    """Assigns importance scores to triples based on entity scores."""
    for triple in all_triples:
        score_sum = 0.0
        for key in ["s", "o", "p"]:
            uri = triple.get(key, "")
            # Check if URI matches the configured dataset
            from ..config.settings import DATASET, RESOURCE_URI_PREFIX, PROPERTY_URI_PREFIX, ONTOLOGY_URI_PREFIX
            if DATASET == "wikidata":
                uri_matches = (uri.startswith("http://www.wikidata.org/") and uri in uri_importance_map)
            else:
                uri_matches = (uri.startswith("http://dbpedia.org/") and uri in uri_importance_map)
            
            if uri_matches:
                score_sum += uri_importance_map[uri]["score"]

        triple["importance"] = _normalize_triple_importance(score_sum)
        triple["_s_name"] = extract_entity_name_from_uri(triple.get("s", ""))
        triple["_o_name"] = extract_entity_name_from_uri(triple.get("o", ""))
        triple["_query_keywords"] = query_keywords
