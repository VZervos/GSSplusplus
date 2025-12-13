from utils.dataset import extract_query_keywords
from utils.dbpedia import (
    lookup_entity_uri, get_uris_from_triples
)
from utils.scoring import compute_importance, assign_importance_scores


def pipeline(query):
    # STEP 1: Initialize pipeline
    print("Starting pipeline...")

    # STEP 2: Extract entities from query
    print("Step 2: Extracting entities...")
    extraction = {"entities": ["GMT Games", "GMT", "board game", "boardgame", "wargame", "tabletop game", "publisher", "game"]}
    print(f"  Extracted {len(extraction['entities'])} entities")

    # STEP 3: Convert entity names to DBpedia URIs
    print("Step 3: Looking up entity URIs...")
    entity_uris = {}
    for entity in extraction["entities"]:
        uri = lookup_entity_uri(entity)
        entity_uris[entity] = uri
        print(f"  {entity} -> {uri}")

    # STEP 4 & 5: Retrieve triples and compute the importance for entity URIs
    print("Step 4 & 5: Retrieving triples and computing importance...")
    all_triples = []
    uri_importance_map = {}

    query_keywords = extract_query_keywords(extraction, query)
    compute_importance(all_triples, entity_uris, uri_importance_map)
    print(f"  Total: {len(all_triples)} triples, {len(uri_importance_map)} URIs with importance scores")
    
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