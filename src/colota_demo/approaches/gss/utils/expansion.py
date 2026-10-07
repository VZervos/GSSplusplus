"""Helper functions for 1-hop expansion of triples."""

MAX_RESOURCES = 50
MAX_BATCHES = 10
BATCH_SIZE = 5
BATCH_LIMIT = 200
PRIORITY_FREQUENCY_MULTIPLIER = 10
PRIORITY_IMPORTANCE_MULTIPLIER = 200
TOP_RESOURCES_DISPLAY = 10
from .dbpedia import (
    get_uris_from_triples, fetch_triples_batch,
    is_predicate_uri, is_resource_uri, extract_entity_name_from_uri
)


def compute_expansion_priorities(
        all_triples: list,
        expansion_uris: set,
        entity_importance: dict
) -> dict:
    """Computes priority scores for expansion URIs."""
    uri_frequency = {}
    uri_priority = {}

    for triple in all_triples:
        for key in ["s", "o", "p"]:
            uri = triple.get(key, "")
            if uri in expansion_uris:
                uri_frequency[uri] = uri_frequency.get(uri, 0) + 1

                max_importance_in_triple = 0
                for other_key in ["s", "o", "p"]:
                    other_uri = triple.get(other_key, "")
                    if other_uri in entity_importance:
                        max_importance_in_triple = max(max_importance_in_triple, entity_importance[other_uri])

                current_priority = uri_priority.get(uri, 0)
                new_priority = (uri_frequency[uri] * PRIORITY_FREQUENCY_MULTIPLIER +
                                max_importance_in_triple * PRIORITY_IMPORTANCE_MULTIPLIER)
                uri_priority[uri] = max(current_priority, new_priority)

    return uri_priority


def expand_to_one_hop(
        all_triples: list,
        original_uris: set,
        entity_importance: dict,
        max_expansion_resources: int = None,
        max_batches: int = None,
        batch_size: int = None,
        batch_limit: int = None
) -> list:
    """Expands triples by fetching 1-hop neighbors of entities found in initial triples."""
    if max_expansion_resources is None:
        max_expansion_resources = MAX_RESOURCES
    if max_batches is None:
        max_batches = MAX_BATCHES
    if batch_size is None:
        batch_size = BATCH_SIZE
    if batch_limit is None:
        batch_limit = BATCH_LIMIT

    expansion_uris = get_uris_from_triples(all_triples)
    expansion_uris = expansion_uris - original_uris

    if not expansion_uris:
        return []

    uri_priority = compute_expansion_priorities(all_triples, expansion_uris, entity_importance)

    expansion_resource_uris = [uri for uri in expansion_uris if is_resource_uri(uri)]
    expansion_predicate_uris = [uri for uri in expansion_uris if is_predicate_uri(uri)]

    expansion_resource_uris.sort(key=lambda u: uri_priority.get(u, 0), reverse=True)
    expansion_predicate_uris.sort(key=lambda u: uri_priority.get(u, 0), reverse=True)

    top_resources = expansion_resource_uris[:TOP_RESOURCES_DISPLAY]
    print(f"    Top prioritized resources: {[extract_entity_name_from_uri(u) for u in top_resources]}")

    expansion_triples = []
    for i in range(0, min(max_expansion_resources, len(expansion_resource_uris)), batch_size):
        if i // batch_size >= max_batches:
            break

        batch_resources = expansion_resource_uris[i:i + batch_size]
        batch_predicates = []

        print(
            f"    Expanding batch {i // batch_size + 1}: {len(batch_resources)} resources, {len(batch_predicates)} predicates")
        batch_triples = fetch_triples_batch(batch_resources, batch_predicates, limit=batch_limit)
        expansion_triples.extend(batch_triples)
        print(f"      Retrieved {len(batch_triples)} triples from this batch")

    return expansion_triples
