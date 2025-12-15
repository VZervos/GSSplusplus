from collections import defaultdict
from utils.dataset import extract_query_keywords
from utils.dbpedia import (
    lookup_entity_uri, get_uris_from_triples
)
from utils.scoring import compute_importance, assign_importance_scores
from utils.parser import extract_entities, filter_keywords
from utils.pruning import prune_bad_uris, clean_triples_from_importance_map
from utils.similarity import compute_similarity_scores
from utils.ranking import (
    compute_final_scores,
    select_subgraph_triples
)
from utils.deduplication import deduplicate_triples

def pipeline(query: str) -> tuple[list, dict]:
    # STEP 1: Initialize pipeline
    print("Starting pipeline...")

    # STEP 2: Extract entities from query
    print("Step 2: Extracting entities...")
    extraction = extract_entities(query)
    entities = extraction.get("entities", [])
    print(f"  Extracted {len(entities)} entities: {entities}")

    # # Expected format: {"entities": [{"name": "GMT_Games", "type": "resource", "importance": 3}, ...]}
    # extraction = {"entities": [
    #     {"name": "GMT_Games", "type": "resource", "importance": 3},
    #     {"name": "GMT", "type": "resource", "importance": 3},
    #     {"name": "Board_Game", "type": "resource", "importance": 2},
    #     {"name": "Boardgame", "type": "resource", "importance": 2},
    #     {"name": "Wargame", "type": "resource", "importance": 2},
    #     {"name": "Tabletop_game", "type": "resource", "importance": 2},
    #     {"name": "publisher", "type": "property", "importance": 3},
    #     {"name": "Game", "type": "resource", "importance": 1}
    # ]}
    # print(f"  Extracted {len(extraction['entities'])} entities")

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
        
        entity_name = (entity_name or "").strip()
        if not entity_name:
            continue
        uri = lookup_entity_uri(entity_name, entity_type)
        entity_uris[entity_name] = uri
        
        entity_importance[uri] = max(entity_importance.get(uri, 0), int(importance or 1))
        print(f"  {entity_name} ({entity_type}) -> {uri} [importance: {importance}]")

    # Deduplicate by URI: group names by URI and fetch once per URI
    uri_to_names = defaultdict(list)
    for name, uri in entity_uris.items():
        uri_to_names[uri].append(name)
    
    uri_map = {}
    for uri, names in uri_to_names.items():
        display_name = max(names, key=len)  # Use longest name as display name
        uri_map[uri] = display_name
    print(f"  Deduplicated: {len(entity_uris)} keywords -> {len(uri_map)} unique URIs")

    # STEP 4 & 5: Retrieve triples and compute the importance for entity URIs
    print("Step 4 & 5: Retrieving triples and computing importance...")
    all_triples = []
    uri_importance_map = {}

    extraction_for_keywords = dict(extraction)
    extraction_for_keywords["entities"] = [
        e.get("name", "") if isinstance(e, dict) else e
        for e in extraction.get("entities", [])
    ]
    query_keywords = extract_query_keywords(extraction_for_keywords, query)
    compute_importance(all_triples, entity_uris, uri_importance_map, entity_importance)
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
    
    # STEP 7: Compute similarity
    print("Step 7: Computing similarity scores...")
    compute_similarity_scores(all_triples, query)
    print(f"  Computed similarity scores for {len(all_triples)} triples")
    
    # STEP 8: Final scoring and ranking
    print("Step 8: Ranking triples...")
    compute_final_scores(all_triples)
    subgraph_triples = select_subgraph_triples(
        all_triples,
        uri_importance_map,
        per_entity_limit=45,
        min_score=0.1,
        max_total=350
    )
    print(f"  Subgraph size: {len(subgraph_triples)} triples")
    
    # STEP 9: Deduplicate equivalent triples
    print("Step 9: Deduplicating triples...")
    subgraph_triples = deduplicate_triples(subgraph_triples)
    print(f"  After deduplication: {len(subgraph_triples)} triples")

    print("Pipeline completed")
    
    return subgraph_triples, uri_importance_map
