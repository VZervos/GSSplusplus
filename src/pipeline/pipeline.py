from config.settings import (
    RANKING_PER_ENTITY_LIMIT,
    RANKING_MIN_SCORE,
    RANKING_MAX_TOTAL
)
from utils.dataset import extract_query_keywords
from utils.deduplication import deduplicate_triples
from utils.entity_processing import parse_entity_extraction, deduplicate_entity_uris
from utils.expansion import expand_to_one_hop
from utils.parser import extract_entities
from utils.pruning import prune_bad_uris, clean_triples_from_importance_map
from utils.ranking import compute_final_scores, select_subgraph_triples
from utils.scoring import compute_importance, assign_importance_scores
from utils.similarity import compute_similarity_scores


def pipeline(query: str) -> tuple[list, dict]:
    """Main pipeline for extracting a knowledge graph subgraph from DBpedia based on a query.
    
    Args:
        query: Natural language query string
    
    Returns:
        Tuple of (selected triples list, URI importance map dict)
    """
    print("Starting pipeline...")

    # STEP 2: Extract entities from query
    print("Step 2: Extracting entities...")
    extraction = extract_entities(query)
    entities = extraction.get("entities", [])
    print(f"  Extracted {len(entities)} entities: {entities}")

    # STEP 3: Convert entity names to DBpedia URIs
    print("Step 3: Looking up entity URIs...")
    entity_uris, entity_importance = parse_entity_extraction(extraction)

    for entity_name, uri in entity_uris.items():
        importance = entity_importance.get(uri, 1)
        entity_type = "resource" if "resource" in uri else ("property" if "property" in uri else "ontology")
        print(f"  {entity_name} ({entity_type}) -> {uri} [importance: {importance}]")

    uri_map = deduplicate_entity_uris(entity_uris)
    print(f"  Deduplicated: {len(entity_uris)} keywords -> {len(uri_map)} unique URIs")

    # STEP 4: Retrieve triples and compute importance for entity URIs
    print("Step 4: Retrieving triples and computing importance...")
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

    # STEP 5: Expand to 1-hop neighbors
    print("Step 5: Expanding to 1-hop neighbors...")
    original_uris = set(entity_uris.values())
    expansion_triples = expand_to_one_hop(all_triples, original_uris, entity_importance)
    all_triples.extend(expansion_triples)
    print(f"    Retrieved {len(expansion_triples)} total additional triples from expansion")
    print(f"  Total after expansion: {len(all_triples)} triples")

    # STEP 6: Prune bad URIs
    print("Step 6: Pruning bad URIs...")
    uri_map, uri_importance_map, all_triples = prune_bad_uris(uri_map, uri_importance_map, all_triples)
    clean_triples_from_importance_map(uri_importance_map)
    print(f"  After pruning: {len(uri_map)} URIs, {len(all_triples)} triples")

    # STEP 7: Assign importance scores to triples
    print("Step 7: Assigning importance scores to triples...")
    assign_importance_scores(all_triples, query_keywords, uri_importance_map)
    print(f"  Assigned importance scores to {len(all_triples)} triples")

    # STEP 8: Compute similarity scores
    print("Step 8: Computing similarity scores...")
    compute_similarity_scores(all_triples, query)
    print(f"  Computed similarity scores for {len(all_triples)} triples")

    # STEP 9: Final scoring and ranking
    print("Step 9: Ranking triples...")
    compute_final_scores(all_triples)
    subgraph_triples = select_subgraph_triples(
        all_triples,
        uri_importance_map,
        per_entity_limit=RANKING_PER_ENTITY_LIMIT,
        min_score=RANKING_MIN_SCORE,
        max_total=RANKING_MAX_TOTAL
    )
    print(f"  Subgraph size: {len(subgraph_triples)} triples")

    # STEP 10: Deduplicate equivalent triples
    print("Step 10: Deduplicating triples...")
    subgraph_triples = deduplicate_triples(subgraph_triples)
    print(f"  After deduplication: {len(subgraph_triples)} triples")

    print("Pipeline completed")
    return subgraph_triples, uri_importance_map
