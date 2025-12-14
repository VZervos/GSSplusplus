def prune_bad_uris(uri_map, uri_importance_map, all_triples):
    """
    Prune URIs with total_degree == 0 OR triples_count == 0.
    Keeps a fallback: if everything is pruned, keep the single best URI by degree.
    """
    # Identify bad URIs
    bad_uris = set()
    good_uris = []
    
    for uri, importance_data in uri_importance_map.items():
        total_degree = importance_data.get("total_degree", 0)
        triples_count = importance_data.get("triples_count", 0)
        
        if total_degree == 0 or triples_count == 0:
            bad_uris.add(uri)
            print(f"  Pruning {uri_map.get(uri, uri)}: total_degree={total_degree}, triples={triples_count}")
        else:
            good_uris.append((uri, total_degree))
    
    # If we have good URIs, remove bad ones
    if good_uris:
        # Remove bad URIs from maps
        good_uri_set = {uri for uri, _ in good_uris}
        pruned_uri_map = {uri: name for uri, name in uri_map.items() if uri in good_uri_set}
        pruned_uri_importance_map = {uri: data for uri, data in uri_importance_map.items() if uri in good_uri_set}
        
        # Remove triples that belong to bad URIs
        # Keep triple if either subject or object is a good URI
        pruned_triples = []
        for triple in all_triples:
            s_uri = triple.get("s", "")
            o_uri = triple.get("o", "")
            # Keep triple if either subject or object is a good URI
            if s_uri in good_uri_set or o_uri in good_uri_set:
                pruned_triples.append(triple)
        
        return pruned_uri_map, pruned_uri_importance_map, pruned_triples
    
    # Fallback: if everything is bad, keep the single best URI by degree
    if uri_importance_map:
        # Sort by total_degree (descending) and keep the best one
        sorted_uris = sorted(
            uri_importance_map.items(),
            key=lambda x: x[1].get("total_degree", 0),
            reverse=True
        )
        best_uri, best_data = sorted_uris[0]
        best_name = uri_map.get(best_uri, best_uri)
        
        print(f"  Fallback: Keeping best URI by degree: {best_name} ({best_uri})")
        
        # Keep only the best URI
        fallback_uri_map = {best_uri: best_name}
        fallback_uri_importance_map = {best_uri: best_data}
        
        # Keep only triples that involve the best URI
        fallback_triples = []
        for triple in all_triples:
            s_uri = triple.get("s", "")
            o_uri = triple.get("o", "")
            if s_uri == best_uri or o_uri == best_uri:
                fallback_triples.append(triple)
        
        return fallback_uri_map, fallback_uri_importance_map, fallback_triples
    
    # Edge case: no URIs at all
    return uri_map, uri_importance_map, all_triples


def clean_triples_from_importance_map(uri_importance_map):
    """Remove stored triples from importance map to save memory."""
    for uri, data in uri_importance_map.items():
        if "triples" in data:
            del data["triples"]

