from utils.dbpedia import extract_entity_name_from_uri


def canonical_predicate(p_uri: str) -> str:
    """
    Normalize DBpedia predicates so dbo:publisher and dbp:publisher
    are treated as the same relation.
    """
    name = extract_entity_name_from_uri(p_uri)
    return name.lower() if name else p_uri.lower()


def deduplicate_triples(triples) -> list:
    """
    Deduplicate triples by (subject, predicate_name, object),
    keeping the highest-scoring one.
    """
    seen = {}

    for t in triples:
        key = (
            t.get("s"),
            canonical_predicate(t.get("p", "")),
            t.get("o"),
        )

        if key not in seen or t.get("final_score", 0.0) > seen[key].get("final_score", 0.0):
            seen[key] = t

    return list(seen.values())
