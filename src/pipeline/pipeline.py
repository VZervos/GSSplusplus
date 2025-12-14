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
    # Expected format: {"entities": [{"name": "GMT_Games", "type": "resource", "importance": 3}, ...]}
    extraction = {"entities": [
        {"name": "GMT_Games", "type": "resource", "importance": 3},
        {"name": "GMT", "type": "resource", "importance": 3},
        {"name": "Board_Game", "type": "resource", "importance": 2},
        {"name": "Boardgame", "type": "resource", "importance": 2},
        {"name": "Wargame", "type": "resource", "importance": 2},
        {"name": "Tabletop_game", "type": "resource", "importance": 2},
        {"name": "publisher", "type": "property", "importance": 3},
        {"name": "Game", "type": "resource", "importance": 1}
    ]}
    print(f"  Extracted {len(extraction['entities'])} entities")

    # STEP 3: Convert entity names to DBpedia URIs
    print("Step 3: Looking up entity URIs...")
    entity_uris = {}
    entity_importance = {}
    
    for entity_obj in extraction["entities"]:
        if isinstance(entity_obj, dict):
            entity_name = entity_obj.get("name", "")
            entity_type = entity_obj.get("type", "resource")
            importance = entity_obj.get("importance", 1)
        else:
            # Legacy format: "resource/EntityName"
            entity_name = entity_obj
            entity_type = None
            importance = 1
        
        uri = lookup_entity_uri(entity_name, entity_type)
        entity_uris[entity_name] = uri
        entity_importance[uri] = importance
        print(f"  {entity_name} ({entity_type}) -> {uri} [importance: {importance}]")

    # STEP 4 & 5: Retrieve triples and compute the importance for entity URIs
    print("Step 4 & 5: Retrieving triples and computing importance...")
    all_triples = []
    uri_importance_map = {}

    query_keywords = extract_query_keywords(extraction, query)
    compute_importance(all_triples, entity_uris, uri_importance_map, entity_importance)
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