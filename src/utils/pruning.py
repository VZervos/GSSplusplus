from collections import Counter

def prune_bad_uris(uri_map, uri_importance_map, all_triples):
    """
    Prune URIs with total_degree == 0 OR triples_count == 0.
    If triples_count is missing in uri_importance_map (new scoring), compute it from all_triples.
    Keeps a fallback: if everything is pruned, keep the single best URI by degree.
    """

    # Build triples_count from all_triples (counts appearances in s/o/p)
    counts = Counter()
    for t in all_triples:
        for k in ("s", "o", "p"):
            u = t.get(k, "")
            if u:
                counts[u] += 1

    bad_uris = set()
    good_uris = []

    for uri, importance_data in uri_importance_map.items():
        total_degree = int(importance_data.get("total_degree", 0) or 0)

        # If scoring already provided triples_count, use it; otherwise compute from triples list
        triples_count = importance_data.get("triples_count", None)
        if triples_count is None:
            triples_count = counts.get(uri, 0)
        triples_count = int(triples_count or 0)

        if total_degree == 0 or triples_count == 0:
            bad_uris.add(uri)
            print(f"  Pruning {uri_map.get(uri, uri)}: total_degree={total_degree}, triples={triples_count}")
        else:
            good_uris.append((uri, total_degree))

    if good_uris:
        good_uri_set = {uri for uri, _ in good_uris}

        pruned_uri_map = {uri: name for uri, name in uri_map.items() if uri in good_uri_set}
        pruned_uri_importance_map = {uri: data for uri, data in uri_importance_map.items() if uri in good_uri_set}

        pruned_triples = []
        for triple in all_triples:
            s_uri = triple.get("s", "")
            o_uri = triple.get("o", "")
            p_uri = triple.get("p", "")
            # Keep triple if any part touches a good URI
            if s_uri in good_uri_set or o_uri in good_uri_set or p_uri in good_uri_set:
                pruned_triples.append(triple)

        return pruned_uri_map, pruned_uri_importance_map, pruned_triples

    # Fallback: keep best URI by degree
    if uri_importance_map:
        sorted_uris = sorted(
            uri_importance_map.items(),
            key=lambda x: int(x[1].get("total_degree", 0) or 0),
            reverse=True
        )
        best_uri, best_data = sorted_uris[0]
        best_name = uri_map.get(best_uri, best_uri)

        print(f"  Fallback: Keeping best URI by degree: {best_name} ({best_uri})")

        fallback_uri_map = {best_uri: best_name}
        fallback_uri_importance_map = {best_uri: best_data}

        fallback_triples = []
        for triple in all_triples:
            if triple.get("s") == best_uri or triple.get("o") == best_uri or triple.get("p") == best_uri:
                fallback_triples.append(triple)

        return fallback_uri_map, fallback_uri_importance_map, fallback_triples

    return uri_map, uri_importance_map, all_triples

def clean_triples_from_importance_map(uri_importance_map):
    """Remove stored triples from importance map to save memory."""
    for uri, data in uri_importance_map.items():
        if "triples" in data:
            del data["triples"]

