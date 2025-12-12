import sys
from services.gemini_client import call_gemini_api
from utils.parser import extract_response_text, extract_entities

def pipeline(query):
    print("STEP 1: Starting pipeline")

    # 2️⃣ Extract entities
    print("STEP 2: Extracting entities...")
    extraction = extract_entities(query)
    print(f"STEP 2: Extracted entities: {extraction}")

    # # 3️⃣ Lookup URI
    # print("STEP 3: Looking up entity URI...")
    # uri = lookup_entity_uri(extraction["entities"][0])
    # print(f"STEP 3: Found URI: {uri}")

    # # 4️⃣ Retrieve triples
    # print("STEP 4: Retrieving candidate triples...")
    # triples = retrieve_candidate_triples(uri)
    # print(f"STEP 4: Retrieved {len(triples)} triples")

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
    answer = "dummy answer"
    return answer

def main():
    print("=" * 60)
    print("🤖 GEMINI SEMANTIC SUMMARIZER")
    print("=" * 60)
    print()
    
    prompt = input("Enter your query: ").strip()
    
    if not prompt:
        print("❌ Error: Query cannot be empty. Exiting...")
        sys.exit(1)

    try:
        
        ####This was a test of just doing a prompt with gemini api.
        #-----------------------------------------------------------
        # response = call_gemini_api(prompt)
        # text = extract_response_text(response)
        #-----------------------------------------------------------

        print("=" * 60)
        print("🤖 GEMINI RESPONSE:")
        print("=" * 60)
        result = pipeline(prompt)
        print(result)
        print("=" * 60)

    except Exception as e:
        print("=" * 60)
        print("❌ ERROR:")
        print("=" * 60)
        print(str(e))
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
