from utils.dbpedia import fetch_triples_with_importance, extract_entity_name_from_uri


def compute_importance(TRIPLES_PER_ENTITY_LIMIT: int, all_triples, entity_uris, uri_importance_map):
    for entity, uri in entity_uris.items():
        print(f"  Processing: {entity}")
        result = fetch_triples_with_importance(uri, TRIPLES_PER_ENTITY_LIMIT)

        total_degree = result["total_degree"]
        normalized_score = min(10.0, 1.0 + (total_degree ** 0.5) / 10.0) if total_degree > 0 else 0.0

        uri_importance_map[uri] = {
            "method": "weighted_degree",
            "score": normalized_score,
            "out_degree": result["out_degree"],
            "in_degree": result["in_degree"],
            "total_degree": total_degree
        }

        print(f"    Retrieved {len(result['triples'])} triples, importance: {normalized_score:.2f}")
        all_triples.extend(result["triples"])


def assign_importance_scores(all_triples, query_keywords, uri_importance_map):
    for triple in all_triples:
        scores = []
        for key in ["s", "o"]:
            uri = triple.get(key, "")
            if uri.startswith("http://dbpedia.org/resource/") and uri in uri_importance_map:
                scores.append(uri_importance_map[uri]["score"])

        triple["importance"] = sum(scores) / len(scores) if scores else 0.0
        triple["_s_name"] = extract_entity_name_from_uri(triple.get("s", ""))
        triple["_o_name"] = extract_entity_name_from_uri(triple.get("o", ""))
        triple["_query_keywords"] = query_keywords
