from collections import defaultdict
from utils.dataset import extract_query_keywords
from utils.dbpedia import (
    lookup_entity_uri, get_uris_from_triples, fetch_triples_batch,
    is_predicate_uri, is_resource_uri, extract_entity_name_from_uri
)
from utils.scoring import compute_importance, assign_importance_scores
from utils.parser import extract_entities, filter_keywords
from utils.pruning import prune_bad_uris, clean_triples_from_importance_map
from utils.similarity import compute_similarity_scores
from utils.ranking import (
    compute_final_scores,
    rank_and_select
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
    
    # Expansion step: Extract URIs from triples and fetch their triples (1-hop expansion)
    print("  Expanding to 1-hop neighbors...")
    expansion_uris = get_uris_from_triples(all_triples)
    # Exclude original query entity URIs to avoid duplicates
    original_uris = set(entity_uris.values())
    expansion_uris = expansion_uris - original_uris
    
    if expansion_uris:
        # Count URI frequency in triples and compute priority scores
        uri_frequency = {}
        uri_priority = {}  # Combined priority: frequency + importance of connected entities
        
        for triple in all_triples:
            for key in ["s", "o", "p"]:
                uri = triple.get(key, "")
                if uri in expansion_uris:
                    uri_frequency[uri] = uri_frequency.get(uri, 0) + 1
                    
                    # Calculate priority: check if this URI is connected to high-importance entities
                    # Priority = frequency + max importance of entities in the same triple
                    max_importance_in_triple = 0
                    for other_key in ["s", "o", "p"]:
                        other_uri = triple.get(other_key, "")
                        if other_uri in entity_importance:
                            max_importance_in_triple = max(max_importance_in_triple, entity_importance[other_uri])
                    
                    # Priority score: frequency * 10 + importance * 200 (importance weighted more, scale 1-5)
                    current_priority = uri_priority.get(uri, 0)
                    new_priority = uri_frequency[uri] * 10 + max_importance_in_triple * 200
                    uri_priority[uri] = max(current_priority, new_priority)
        
        # Separate into resources and predicates, sorted by priority
        expansion_resource_uris = [uri for uri in expansion_uris if is_resource_uri(uri)]
        expansion_predicate_uris = [uri for uri in expansion_uris if is_predicate_uri(uri)]
        
        # Sort by priority (highest first) and limit to top ones
        expansion_resource_uris.sort(key=lambda u: uri_priority.get(u, 0), reverse=True)
        expansion_predicate_uris.sort(key=lambda u: uri_priority.get(u, 0), reverse=True)
        
        # Limit expansion to avoid query size issues (smaller batches)
        # Process in smaller batches to avoid 413 errors
        batch_size = 20  # Process 20 resources at a time
        expansion_triples = []
        
        # Show top prioritized URIs
        top_resources = expansion_resource_uris[:10]
        print(f"    Top prioritized resources: {[extract_entity_name_from_uri(u) for u in top_resources]}")
        
        # Increase expansion limit to get more triples
        max_expansion_resources = 100  # Increased from 50
        for i in range(0, min(max_expansion_resources, len(expansion_resource_uris)), batch_size):
            batch_resources = expansion_resource_uris[i:i+batch_size]
            batch_predicates = expansion_predicate_uris[:10] if i == 0 else []  # Only include predicates in first batch
            
            print(f"    Expanding batch {i//batch_size + 1}: {len(batch_resources)} resources, {len(batch_predicates)} predicates")
            batch_limit = 500 * max(len(batch_resources), len(batch_predicates), 1)  # Increased from 300
            batch_triples = fetch_triples_batch(batch_resources, batch_predicates, limit=batch_limit)
            expansion_triples.extend(batch_triples)
            print(f"      Retrieved {len(batch_triples)} triples from this batch")
        
        all_triples.extend(expansion_triples)
        print(f"    Retrieved {len(expansion_triples)} total additional triples from expansion")
        print(f"  Total after expansion: {len(all_triples)} triples")
    
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
    top_triples = rank_and_select(all_triples, k=1000)
    print(f"  Selected top {len(top_triples)} triples")    
    
    # STEP 9: Deduplicate equivalent triples
    print("Step 9: Deduplicating triples...")
    top_triples = deduplicate_triples(top_triples)
    print(f"  After deduplication: {len(top_triples)} triples")

    print("Pipeline completed")
    
    return top_triples, uri_importance_map
