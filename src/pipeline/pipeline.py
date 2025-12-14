from collections import defaultdict
from utils.dataset import extract_query_keywords
from utils.dbpedia import (
    lookup_entity_uri, get_uris_from_triples
)
from utils.scoring import compute_importance, assign_importance_scores
from utils.parser import extract_entities, filter_keywords
from utils.pruning import prune_bad_uris, clean_triples_from_importance_map


def pipeline(query):
    # STEP 1: Initialize pipeline
    print("Starting pipeline...")

    # STEP 2: Extract entities from query
    print("Step 2: Extracting entities...")
    extraction = extract_entities(query)

    raw_keywords = extraction.get("entities", []) + extraction.get("synonyms", [])
    keywords = filter_keywords(raw_keywords)

    print(f"  Extracted {len(extraction.get('entities', []))} entities, {len(extraction.get('synonyms', []))} synonyms")
    print(f"  Final keywords for DBpedia lookup: {keywords}")

    extraction["entities"] = keywords if keywords else ["UnknownEntity"]


    # STEP 3: Convert entity names to DBpedia URIs
    print("Step 3: Looking up entity URIs...")
    entity_uri_map = {}
    for entity in extraction["entities"]:
        uri = lookup_entity_uri(entity)
        entity_uri_map[entity] = uri
        print(f"  {entity} -> {uri}")

    # Deduplicate by URI: group names by URI and fetch once per URI
    uri_to_names = defaultdict(list)
    for name, uri in entity_uri_map.items():
        uri_to_names[uri].append(name)
    
    uri_map = {}
    for uri, names in uri_to_names.items():
        display_name = max(names, key=len)  # Use longest name as display name
        uri_map[uri] = display_name
    print(f"  Deduplicated: {len(entity_uri_map)} keywords -> {len(uri_map)} unique URIs")

    # STEP 4 & 5: Retrieve triples and compute the importance for entity URIs
    print("Step 4 & 5: Retrieving triples and computing importance...")
    all_triples = []
    uri_importance_map = {}

    query_keywords = extract_query_keywords(extraction, query)
    compute_importance(all_triples, uri_map, uri_importance_map)
    print(f"  Total: {len(all_triples)} triples, {len(uri_importance_map)} URIs with importance scores")
    
    print("Pruning bad URIs...")
    uri_map, uri_importance_map, all_triples = prune_bad_uris(uri_map, uri_importance_map, all_triples)
    clean_triples_from_importance_map(uri_importance_map)  # Clean up stored triples
    print(f"  After pruning: {len(uri_map)} URIs, {len(all_triples)} triples")
    
    # STEP 6: Assign importance scores to triples based on their URIs
    print("Step 6: Assigning importance scores to triples...")
    get_uris_from_triples(all_triples)
    assign_importance_scores(all_triples, query_keywords, uri_importance_map)
    print(f"  Assigned importance scores to {len(all_triples)} triples")
    
    # STEP 7: Compute similarity (not yet implemented)
    # STEP 8: Score & rank (not yet implemented)
    # STEP 9: Select top K (not yet implemented)
    # STEP 10: Verbalize result (not yet implemented)
    
    print("Pipeline completed")
    
    return {"triples": all_triples, "importance_map": uri_importance_map}
