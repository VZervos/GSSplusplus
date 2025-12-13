from utils.dbpedia import lookup_entity_uri, retrieve_candidate_triples, fetch_triples_from_sparql


def pipeline(query):
    print("STEP 1: Starting pipeline")

    # 2️⃣ Extract entities
    print("STEP 2: Extracting entities...")
    # extraction = extract_entities(query) TODO: Remove
    extraction = {"entities": query.split()}
    print(f"STEP 2: Extracted entities: {extraction}")

    all_triples = []
    for entity in extraction["entities"]:
        print(f"Processing entity: {entity}")
        # # 3️⃣ Lookup URI
        print("STEP 3: Looking up entity URI...")
        uri = lookup_entity_uri(entity)
        print(f"Found URI: {uri}")

        # # 4️⃣ Retrieve triples
        print("STEP 4: Retrieving candidate triples...")
        sparql_query = retrieve_candidate_triples(uri)
        print(f"SPARQL query generated")

        # Fetch actual triples from SPARQL endpoint
        uri_triples = fetch_triples_from_sparql(sparql_query)
        print(f"Retrieved {len(uri_triples)} triples for entity {entity}")
        all_triples.extend(uri_triples)

    # # 5️⃣ Compute importance (dummy)
    # print("STEP 5: Computing importance scores...")
    # for t in triples:
    #     t["importance"] = compute_importance(t)
    # print("STEP 5: Importance scores computed")

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
    return all_triples
