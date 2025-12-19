import time

import requests

from config.settings import (
    DBPEDIA_MAX_RETRIES,
    DBPEDIA_INITIAL_TIMEOUT,
    DBPEDIA_MAX_TIMEOUT,
    DBPEDIA_MAX_UNION_CLAUSES,
    DBPEDIA_DEFAULT_TRIPLE_LIMIT
)


def execute_sparql_query_with_retry(sparql_query: str, max_retries: int = None, initial_timeout: int = None,
                                    max_timeout: int = None) -> dict:
    """
    Executes a SPARQL query with retry logic and exponential backoff.
    Only retries on timeouts (transient issues), not on other errors.
    
    Args:
        sparql_query: The SPARQL query string
        max_retries: Maximum number of retry attempts (default: from settings)
        initial_timeout: Initial timeout in seconds (default: from settings)
        max_timeout: Maximum timeout in seconds (default: from settings)
    
    Returns:
        dict: JSON response from SPARQL endpoint, or None if all retries fail
    """
    if max_retries is None:
        max_retries = DBPEDIA_MAX_RETRIES
    if initial_timeout is None:
        initial_timeout = DBPEDIA_INITIAL_TIMEOUT
    if max_timeout is None:
        max_timeout = DBPEDIA_MAX_TIMEOUT
    sparql_endpoint = "https://dbpedia.org/sparql"
    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0"
    }

    data = {"query": sparql_query, "format": "json"}

    for attempt in range(max_retries):
        timeout = min(initial_timeout * (2 ** attempt), max_timeout)

        try:
            if attempt > 0:
                wait_time = 2 ** attempt
                print(f"    Retrying after {wait_time}s...")
                time.sleep(wait_time)

            response = requests.post(sparql_endpoint, headers=headers, data=data, timeout=timeout)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"    Timeout ({timeout}s), retrying...")
                continue
            else:
                print(f"    Timeout after {max_retries} attempts")
                return None

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 405:
                print(f"    Error 405: Query too large or method not allowed. Try reducing batch size.")
            elif e.response.status_code == 500:
                print(f"    Error 500: Server error. Query may be too complex.")
            elif e.response.status_code == 413:
                print(f"    Error 413: Request entity too large. Try reducing batch size.")
            else:
                print(f"    HTTP Error {e.response.status_code}: {str(e)}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"    Error: {str(e)}")
            return None

    return None


def _format_entity_name(name: str) -> str:
    """Formats entity name for DBpedia URI (handles spaces and underscores)."""
    return "_".join(part for part in name.replace(" ", "_").split("_") if part)


def lookup_entity_uri(entity: str, entity_type: str = None) -> str:
    """Converts an entity name to a DBpedia URI.
    
    Args:
        entity: Entity name (e.g., "GMT_Games" or "resource/GMT_Games")
        entity_type: Type of entity ("resource", "property", "ontology") - optional if entity contains type prefix
    
    Returns:
        DBpedia URI string
    """
    if "/" in entity and entity_type is None:
        parts = entity.split("/", 1)
        if len(parts) == 2:
            uri_type, name = parts
            return f"http://dbpedia.org/{uri_type}/{_format_entity_name(name)}"

    if entity_type == "property":
        uri_type = "property"
    elif entity_type == "ontology":
        uri_type = "ontology"
    else:
        uri_type = "resource"

    return f"http://dbpedia.org/{uri_type}/{_format_entity_name(entity)}"


def is_predicate_uri(uri: str) -> bool:
    """Checks if a URI is a predicate (property or ontology)."""
    return uri.startswith("http://dbpedia.org/property/") or uri.startswith("http://dbpedia.org/ontology/")


def is_resource_uri(uri: str) -> bool:
    """Checks if a URI is a resource."""
    return uri.startswith("http://dbpedia.org/resource/")


def extract_entity_name_from_uri(uri: str) -> str:
    """Extracts a readable entity name from a DBpedia URI."""
    if not uri or not uri.startswith("http://dbpedia.org/"):
        return None

    if uri.startswith("http://dbpedia.org/resource/"):
        return uri.replace("http://dbpedia.org/resource/", "").replace("_", " ").lower()
    elif uri.startswith("http://dbpedia.org/property/"):
        return uri.replace("http://dbpedia.org/property/", "").replace("_", " ").lower()
    elif uri.startswith("http://dbpedia.org/ontology/"):
        return uri.replace("http://dbpedia.org/ontology/", "").replace("_", " ").lower()

    return None


_NOISY_PREDICATES = [
    "rdf:type",
    "rdfs:label",
    "dbo:wikiPageID",
    "<http://dbpedia.org/ontology/wikiPageWikiLink>",
    "<http://xmlns.com/foaf/0.1/name>",
    "<http://www.w3.org/2000/01/rdf-schema#comment>",
    "<http://dbpedia.org/ontology/wikiPageRedirects>",
    "<http://dbpedia.org/ontology/wikiPageDisambiguates>",
    "<http://dbpedia.org/property/wikiPageUsesTemplate>",
    "<http://purl.org/linguistics/gold/hypernym>",
    "<http://purl.org/dc/terms/subject>",
    "<http://www.w3.org/2000/01/rdf-schema#seeAlso>",
    "<http://www.w3.org/2002/07/owl#differentFrom>"
]


def _parse_triples_from_bindings(bindings: list) -> list:
    """Helper to parse triples from SPARQL bindings."""
    triples = []
    for binding in bindings:
        triple = {var: binding[var].get("value", "") for var in ["s", "p", "o"] if var in binding}
        if triple:
            triples.append(triple)
    return triples


def generate_degree_query(uri: str) -> str:
    """Generates a SPARQL query to compute both out-degree and in-degree for a URI.
    
    For predicates: counts how many triples use this predicate.
    For resources: counts connections as subject (out-degree) and object (in-degree).
    """
    predicate_filter = _get_predicate_filter_string()

    if is_predicate_uri(uri):
        sparql_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        
        SELECT 
            (COUNT(DISTINCT ?s) AS ?out_degree)
            (COUNT(DISTINCT ?o) AS ?in_degree)
        WHERE {{
            ?s <{uri}> ?o .
            FILTER(isIRI(?s))
            FILTER(isIRI(?o))
            FILTER(STRSTARTS(STR(?s), "http://dbpedia.org/resource/"))
            FILTER(STRSTARTS(STR(?o), "http://dbpedia.org/resource/"))
        }}
        """
    else:
        sparql_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        
        SELECT 
            (COUNT(DISTINCT ?o_out) AS ?out_degree)
            (COUNT(DISTINCT ?s_in) AS ?in_degree)
        WHERE {{
            OPTIONAL {{
                <{uri}> ?p_out ?o_out .
                FILTER(isIRI(?o_out))
                FILTER(STRSTARTS(STR(?o_out), "http://dbpedia.org/resource/"))
                FILTER(?p_out NOT IN (
                    {predicate_filter}
                ))
            }}
            OPTIONAL {{
                ?s_in ?p_in <{uri}> .
                FILTER(isIRI(?s_in))
                FILTER(STRSTARTS(STR(?s_in), "http://dbpedia.org/resource/"))
                FILTER(?p_in NOT IN (
                    {predicate_filter}
                ))
            }}
        }}
        """
    return sparql_query


def compute_weighted_degree(uri: str) -> tuple:
    """Computes weighted degree centrality for a URI. Returns (out_degree, in_degree, total_degree)."""
    result_data = execute_sparql_query_with_retry(generate_degree_query(uri), max_retries=2, initial_timeout=45)

    out_degree = in_degree = 0
    if result_data and "results" in result_data and "bindings" in result_data["results"]:
        binding = result_data["results"]["bindings"][0] if result_data["results"]["bindings"] else {}
        out_degree = int(binding.get("out_degree", {}).get("value", "0"))
        in_degree = int(binding.get("in_degree", {}).get("value", "0"))

    return (out_degree, in_degree, out_degree + in_degree)


def _get_predicate_filter_string() -> str:
    """Returns formatted predicate filter string for SPARQL queries."""
    return ",\n                ".join(_NOISY_PREDICATES)


def _build_resource_pattern(resource: str, as_subject: bool = True, predicate: str = None) -> str:
    """Builds a SPARQL pattern for a resource as subject or object."""
    if as_subject:
        if predicate:
            return f"""
                {{
                    <{resource}> <{predicate}> ?o .
                    BIND(<{resource}> AS ?s)
                    BIND(<{predicate}> AS ?p)
                    FILTER(isIRI(?o))
                    FILTER(STRSTARTS(STR(?o), "http://dbpedia.org/resource/"))
                }}"""
        else:
            predicate_filter = _get_predicate_filter_string()
            return f"""
            {{
                <{resource}> ?p ?o .
                BIND(<{resource}> AS ?s)
                FILTER(isIRI(?o))
                FILTER(STRSTARTS(STR(?o), "http://dbpedia.org/resource/"))
                FILTER(?p NOT IN (
                    {predicate_filter}
                ))
            }}"""
    else:
        if predicate:
            return f"""
                {{
                    ?s <{predicate}> <{resource}> .
                    BIND(<{predicate}> AS ?p)
                    BIND(<{resource}> AS ?o)
                    FILTER(isIRI(?s))
                    FILTER(STRSTARTS(STR(?s), "http://dbpedia.org/resource/"))
                }}"""
        else:
            predicate_filter = _get_predicate_filter_string()
            return f"""
            {{
                ?s ?p <{resource}> .
                BIND(<{resource}> AS ?o)
                FILTER(isIRI(?s))
                FILTER(STRSTARTS(STR(?s), "http://dbpedia.org/resource/"))
                FILTER(?p NOT IN (
                    {predicate_filter}
                ))
            }}"""


def _build_predicate_pattern(predicate: str) -> str:
    """Builds a SPARQL pattern for a predicate."""
    return f"""
            {{
                ?s <{predicate}> ?o .
                BIND(<{predicate}> AS ?p)
                FILTER(isIRI(?s))
                FILTER(isIRI(?o))
                FILTER(STRSTARTS(STR(?s), "http://dbpedia.org/resource/"))
                FILTER(STRSTARTS(STR(?o), "http://dbpedia.org/resource/"))
            }}"""


def generate_batch_retrieve_query(resource_uris: list, predicate_uris: list, limit: int = None) -> str:
    """Generates a SPARQL query to retrieve triples using resources and/or predicates.
    
    Args:
        resource_uris: List of resource URIs (can be empty)
        predicate_uris: List of predicate URIs (can be empty)
        limit: Maximum number of triples to return
    
    Note: Limits query complexity to avoid 405/500 errors from DBpedia.
    When both resources and predicates are present, prioritizes resource+predicate combinations
    and limits the number of UNION clauses.
    """
    if limit is None:
        limit = DBPEDIA_DEFAULT_TRIPLE_LIMIT

    if not resource_uris and not predicate_uris:
        return None

    union_patterns = []

    if resource_uris and predicate_uris:
        max_resources = min(len(resource_uris), 10)
        max_predicates = min(len(predicate_uris), 5)

        # Case 1: Resources with predicates (prioritized)
        for resource in resource_uris[:max_resources]:
            for predicate in predicate_uris[:max_predicates]:
                if len(union_patterns) >= DBPEDIA_MAX_UNION_CLAUSES:
                    break
                union_patterns.append(_build_resource_pattern(resource, as_subject=True, predicate=predicate))
                if len(union_patterns) >= DBPEDIA_MAX_UNION_CLAUSES:
                    break
                union_patterns.append(_build_resource_pattern(resource, as_subject=False, predicate=predicate))
            if len(union_patterns) >= DBPEDIA_MAX_UNION_CLAUSES:
                break

        # Case 2: Add resources with any predicate if room available
        if len(union_patterns) < DBPEDIA_MAX_UNION_CLAUSES:
            remaining = DBPEDIA_MAX_UNION_CLAUSES - len(union_patterns)
            for resource in resource_uris[:min(len(resource_uris), remaining // 2)]:
                if len(union_patterns) >= DBPEDIA_MAX_UNION_CLAUSES:
                    break
                union_patterns.append(_build_resource_pattern(resource, as_subject=True))
                if len(union_patterns) >= DBPEDIA_MAX_UNION_CLAUSES:
                    break
                union_patterns.append(_build_resource_pattern(resource, as_subject=False))
    elif resource_uris:
        max_resources = min(len(resource_uris), DBPEDIA_MAX_UNION_CLAUSES // 2)
        for resource in resource_uris[:max_resources]:
            union_patterns.append(_build_resource_pattern(resource, as_subject=True))
            union_patterns.append(_build_resource_pattern(resource, as_subject=False))
    else:
        max_predicates = min(len(predicate_uris), DBPEDIA_MAX_UNION_CLAUSES)
        for predicate in predicate_uris[:max_predicates]:
            union_patterns.append(_build_predicate_pattern(predicate))

    if not union_patterns:
        return None

    union_clauses = "\n            UNION".join(union_patterns)

    return f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT ?s ?p ?o
    WHERE {{
        {union_clauses}
    }}
    LIMIT {limit}
    """


def fetch_triples_batch(resource_uris: list, predicate_uris: list, limit: int = None) -> list:
    """Fetches triples in batch where subject/object is any resource or predicate is any predicate.
    
    Args:
        resource_uris: List of resource URIs
        predicate_uris: List of predicate URIs
        limit: Maximum number of triples to return (default: from settings)
    
    Returns:
        List of triples
    """
    if limit is None:
        limit = DBPEDIA_DEFAULT_TRIPLE_LIMIT

    if not resource_uris and not predicate_uris:
        return []

    query = generate_batch_retrieve_query(resource_uris, predicate_uris, limit)
    if not query:
        return []

    result_data = execute_sparql_query_with_retry(query, max_retries=2, initial_timeout=45)
    if result_data and "results" in result_data and "bindings" in result_data["results"]:
        return _parse_triples_from_bindings(result_data["results"]["bindings"])

    return []


def get_uris_from_triples(all_triples: list) -> set:
    """Extracts all DBpedia URIs (resources, properties, ontologies) from triples."""
    all_uris_in_triples = set()
    for triple in all_triples:
        for key in ["s", "o", "p"]:
            uri = triple.get(key, "")
            if uri.startswith("http://dbpedia.org/"):
                all_uris_in_triples.add(uri)
    return all_uris_in_triples
