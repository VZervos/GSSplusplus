"""Answer-centered GSS pipeline for hallucination detection."""

from __future__ import annotations

from config.settings import (
    RANKING_PER_ENTITY_LIMIT,
    RANKING_MIN_SCORE,
    RANKING_MAX_TOTAL,
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
from utils.dbpedia import lookup_entity_uri
from services.llm_client import call_llm_api, extract_response_text


ANSWER_ENTITY_PROMPT = """Extract the main answer entity from the LLM answer below.
Return ONLY valid JSON (no markdown):
{{"name": "...", "type": "resource", "importance": 5}}

Use a DBpedia-friendly Title_Case_With_Underscores name when possible
(e.g., "New York" -> "New_York", "United States" -> "United_States").
If the answer is already a short entity string, keep it.

Question: {question}
LLM answer: {answer}
"""


def extract_answer_entity(question: str, answer: str) -> dict:
    """Extract a single resource entity from a free-text LLM answer."""
    answer = (answer or "").strip()
    if not answer:
        return {"name": "", "type": "resource", "importance": 5}

    # Prefer a cheap heuristic for short answers
    if len(answer.split()) <= 6 and "\n" not in answer:
        name = answer.strip().strip(".").strip('"').strip("'")
        name = name.replace(" ", "_")
        return {"name": name, "type": "resource", "importance": 5}

    response = call_llm_api(
        ANSWER_ENTITY_PROMPT.format(question=question, answer=answer),
        timeout=120,
    )
    text = extract_response_text(response).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    import json

    try:
        data = json.loads(text)
        name = (data.get("name") or answer).strip()
        return {
            "name": name.replace(" ", "_"),
            "type": data.get("type") or "resource",
            "importance": int(data.get("importance") or 5),
        }
    except Exception:
        return {
            "name": answer.splitlines()[0].strip().replace(" ", "_"),
            "type": "resource",
            "importance": 5,
        }


def pipeline_answer_centered(
    query: str,
    answer_text: str,
    answer_entity: dict | None = None,
    include_query_entities: bool = True,
) -> tuple[list, dict, dict]:
    """Build an ACE subgraph centered on the LLM answer entity.

    Returns:
        (subgraph_triples, uri_importance_map, meta)
        meta includes answer_entity, answer_uri, entity_uris
    """
    print("Starting answer-centered pipeline...")
    if answer_entity is None:
        answer_entity = extract_answer_entity(query, answer_text)
    print(f"  Answer entity seed: {answer_entity}")

    answer_name = (answer_entity.get("name") or "").strip()
    if not answer_name:
        return [], {}, {
            "answer_entity": answer_entity,
            "answer_uri": None,
            "entity_uris": {},
            "error": "empty_answer_entity",
        }

    answer_uri = lookup_entity_uri(answer_name, answer_entity.get("type") or "resource")
    entity_uris = {answer_name: answer_uri}
    entity_importance = {answer_uri: int(answer_entity.get("importance") or 5)}

    if include_query_entities:
        print("  Extracting supporting entities from the question...")
        extraction = extract_entities(query)
        q_uris, q_imp = parse_entity_extraction(extraction)
        for name, uri in q_uris.items():
            if uri == answer_uri:
                continue
            entity_uris[name] = uri
            # Keep answer entity dominant
            entity_importance[uri] = max(
                entity_importance.get(uri, 0),
                min(int(q_imp.get(uri, 1)), 3),
            )
    else:
        extraction = {"entities": [answer_entity]}

    uri_map = deduplicate_entity_uris(entity_uris)
    print(f"  Seed URIs: {len(uri_map)} (answer={answer_uri})")

    all_triples: list = []
    uri_importance_map: dict = {}

    extraction_for_keywords = {
        "entities": [
            e.get("name", "") if isinstance(e, dict) else e
            for e in extraction.get("entities", [])
        ]
    }
    if answer_name not in extraction_for_keywords["entities"]:
        extraction_for_keywords["entities"].insert(0, answer_name)

    query_keywords = extract_query_keywords(extraction_for_keywords, query)
    compute_importance(all_triples, entity_uris, uri_importance_map, entity_importance)
    # Ensure answer URI keeps top importance even if retrieval is sparse
    if answer_uri in uri_importance_map and isinstance(uri_importance_map[answer_uri], dict):
        uri_importance_map[answer_uri]["score"] = max(
            float(uri_importance_map[answer_uri].get("score", 0.0)),
            1.0,
        )
    else:
        uri_importance_map[answer_uri] = {
            "method": "answer_seed",
            "score": 1.0,
            "base_score": 1.0,
            "importance_weight": int(answer_entity.get("importance") or 5),
            "out_degree": 0,
            "in_degree": 0,
            "total_degree": 0,
        }

    print("  Expanding to 1-hop neighbors around answer-centered triples...")
    original_uris = {answer_uri}
    expansion_triples = expand_to_one_hop(all_triples, original_uris, entity_importance)
    all_triples.extend(expansion_triples)

    uri_map, uri_importance_map, all_triples = prune_bad_uris(
        uri_map, uri_importance_map, all_triples
    )
    clean_triples_from_importance_map(uri_importance_map)

    assign_importance_scores(all_triples, query_keywords, uri_importance_map)
    compute_similarity_scores(all_triples, query)
    compute_final_scores(all_triples)
    subgraph_triples = select_subgraph_triples(
        all_triples,
        uri_importance_map,
        per_entity_limit=RANKING_PER_ENTITY_LIMIT,
        min_score=RANKING_MIN_SCORE,
        max_total=RANKING_MAX_TOTAL,
    )
    subgraph_triples = deduplicate_triples(subgraph_triples)
    print(f"  Answer-centered subgraph size: {len(subgraph_triples)} triples")

    meta = {
        "answer_entity": answer_entity,
        "answer_uri": answer_uri,
        "entity_uris": entity_uris,
        "uri_map": uri_map,
    }
    return subgraph_triples, uri_importance_map, meta
