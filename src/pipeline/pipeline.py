from utils.dbpedia import lookup_entity_uri, generate_retrieve_query, fetch_triples_from_sparql, compute_importance


def pipeline(query):
    print("STEP 1: Starting pipeline")

    print("STEP 2: Extracting entities...")
    # extraction = extract_entities(query) TODO: Remove
    extraction = {"entities": query.split()}
    extraction["entities"][-1] = extraction["entities"][-1][:-1]
    print(f"STEP 2: Extracted entities: {extraction}")

    print("STEP 3: Looking up entity URIs...")
    entity_uris = {}
    for entity in extraction["entities"]:
        uri = lookup_entity_uri(entity)
        entity_uris[entity] = uri
        print(f"  {entity} -> {uri}")

    # 5️⃣ Compute importance scores for entity URIs (before retrieving triples)
    print("STEP 5: Computing importance scores for entities...")
    uri_importance_map = {}  # Store importance scores for each URI
    
    for entity, uri in entity_uris.items():
        if uri not in uri_importance_map:
            print(f"  Computing importance for {entity} ({uri})...")
            importance = compute_importance(uri)
            uri_importance_map[uri] = importance
            print(f"    Method: {importance['method']}, Score: {importance['score']:.4f}")
            if importance['method'] == 'weighted_degree':
                print(f"    Out-degree: {importance['out_degree']}, In-degree: {importance['in_degree']}, Total: {importance['total_degree']}")
    
    print(f"STEP 5: Computed importance scores for {len(uri_importance_map)} entity URIs")
    for uri, importance in uri_importance_map.items():
        print(f"  {uri}: {importance['method']} score = {importance['score']:.4f}")

    return # TODO Remove
    # STEP 4: Retrieve candidate triples (moved after importance computation)
    print("STEP 4: Retrieving candidate triples...")
    all_triples = []
    
    for entity, uri in entity_uris.items():
        print(f"  Retrieving triples for {entity} ({uri})...")
        uri_triples = fetch_triples_from_sparql(uri)
        print(f"    Retrieved {len(uri_triples)} triples for entity {entity}")
        all_triples.extend(uri_triples)
    
    print(f"STEP 4: Retrieved {len(all_triples)} total triples")
    
    # # 6️⃣ Compute similarity (dummy)
    # print("STEP 6: Computing similarity scores...")
    # for t in triples:
    #     t["similarity"] = compute_similarity(query, t)
    # print("STEP 6: Similarity scores computed")

    # # 7️⃣ Score & rank (dummy)
    # print("STEP 7: Scoring and ranking triples...")
    # ranked = score_and_rank_triples(triples)
    # print("STEP 7: Triples scored and ranked")

    # # 8️⃣ Select top K
    # print("STEP 8: Selecting top K triples...")
    # top_k = select_top_k(ranked)
    # print(f"STEP 8: Selected {len(top_k)} top triples")

    # # 9️⃣ Verbalize result
    # print("STEP 9: Verbalizing answer...")
    # answer = verbalize_answer(top_k)
    # print("STEP 9: Answer verbalized")

    # print("STEP 9: Using dummy answer")

    print("Pipeline completed successfully")
    
    # Store importance map in a way that can be accessed later
    # We'll return it along with triples
    return {
        "triples": all_triples,
        "importance_map": uri_importance_map
    }
