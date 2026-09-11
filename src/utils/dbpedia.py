import time

import requests

from config.settings import (
    DATASET,
    SPARQL_ENDPOINT,
    RESOURCE_URI_PREFIX,
    PROPERTY_URI_PREFIX,
    ONTOLOGY_URI_PREFIX,
    KG_MAX_RETRIES,
    KG_INITIAL_TIMEOUT,
    KG_MAX_TIMEOUT,
    KG_MAX_UNION_CLAUSES,
    KG_DEFAULT_TRIPLE_LIMIT,
    # Backward compatibility
    DBPEDIA_MAX_RETRIES,
    DBPEDIA_INITIAL_TIMEOUT,
    DBPEDIA_MAX_TIMEOUT,
    DBPEDIA_MAX_UNION_CLAUSES,
    DBPEDIA_DEFAULT_TRIPLE_LIMIT
)


def execute_sparql_query_with_retry(sparql_query: str, max_retries: int = None, initial_timeout: int = None,
                                    max_timeout: int = None) -> dict:
    """Executes a SPARQL query with retry logic and exponential backoff."""
    if max_retries is None:
        max_retries = KG_MAX_RETRIES
    if initial_timeout is None:
        initial_timeout = KG_INITIAL_TIMEOUT
    if max_timeout is None:
        max_timeout = KG_MAX_TIMEOUT
    
    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded",
        # Wikidata requires an identifying User-Agent (anonymous/generic agents get 403).
        "User-Agent": "GSSplusplus-research/1.0 (hallucination-pilot; academic use)",
    }

    data = {"query": sparql_query, "format": "json"}

    for attempt in range(max_retries):
        timeout = min(initial_timeout * (2 ** attempt), max_timeout)

        try:
            if attempt > 0:
                wait_time = 2 ** attempt
                print(f"    Retrying after {wait_time}s...")
                time.sleep(wait_time)

            response = requests.post(SPARQL_ENDPOINT, headers=headers, data=data, timeout=timeout)
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
            status = e.response.status_code if e.response is not None else None
            # Wikidata often returns 403/429 under load — back off and retry.
            if status in (403, 429) and attempt < max_retries - 1:
                wait_time = max(5, 2 ** (attempt + 2))
                print(
                    f"    HTTP {status} (rate limit). Waiting {wait_time}s before retry..."
                )
                time.sleep(wait_time)
                continue
            if status == 405:
                print(f"    Error 405: Query too large or method not allowed. Try reducing batch size.")
            elif status == 500:
                print(f"    Error 500: Server error. Query may be too complex.")
            elif status == 413:
                print(f"    Error 413: Request entity too large. Try reducing batch size.")
            else:
                print(f"    HTTP Error {status}: {str(e)}")
            if attempt < max_retries - 1 and status in (500, 502, 503):
                continue
            return None
        except requests.exceptions.RequestException as e:
            print(f"    Error: {str(e)}")
            return None

    return None


def _format_entity_name(name: str) -> str:
    """Formats entity name for URI generation."""
    if DATASET == "wikidata":
        # For Wikidata, entity names should be Q/P numbers (e.g., "Q30" or "P31")
        # If it's already a Q/P number, return as is
        name_upper = name.upper().strip()
        if name_upper.startswith("Q") or name_upper.startswith("P"):
            # Check if it's a valid format (Q/P followed by digits)
            if len(name_upper) > 1 and name_upper[1:].isdigit():
                return name_upper
        # Otherwise, try to extract Q/P number from the name
        # This is a fallback - ideally the LLM should provide Q/P numbers
        return "_".join(part for part in name.replace(" ", "_").split("_") if part)
    else:
        # DBpedia: Title_Case_With_Underscores
        return "_".join(part for part in name.replace(" ", "_").split("_") if part)


def lookup_entity_uri(entity: str, entity_type: str = None) -> str:
    """Converts an entity name to a URI based on the configured dataset."""
    entity = (entity or "").strip()
    if not entity:
        return None

    # Bare Wikidata Q/P ids from LAMA or the extractor
    if DATASET == "wikidata":
        if entity.startswith("http://www.wikidata.org/"):
            return entity
        if entity.startswith("wd:"):
            return RESOURCE_URI_PREFIX + entity[3:]
        if len(entity) > 1 and entity[0] in "qQ" and entity[1:].isdigit():
            return f"{RESOURCE_URI_PREFIX}Q{entity[1:]}"
        if entity_type == "property" and len(entity) > 1 and entity[0] in "pP" and entity[1:].isdigit():
            return f"{PROPERTY_URI_PREFIX}P{entity[1:]}"

    # If entity already contains a full URI, try to parse it
    if "/" in entity and entity_type is None:
        parts = entity.split("/", 1)
        if len(parts) == 2:
            uri_type, name = parts
            if DATASET == "wikidata":
                # For Wikidata, if it's already a full URI, return as is
                if entity.startswith("http://www.wikidata.org/"):
                    return entity
                # Otherwise construct from parts
                formatted = _format_entity_name(name)
                if uri_type == "entity" or uri_type == "resource":
                    return f"{RESOURCE_URI_PREFIX}{formatted}"
                elif uri_type == "property" or uri_type == "prop/direct":
                    return f"{PROPERTY_URI_PREFIX}{formatted}"
            else:
                # DBpedia
                return f"http://dbpedia.org/{uri_type}/{_format_entity_name(name)}"

    # Determine URI type
    if entity_type == "property":
        uri_prefix = PROPERTY_URI_PREFIX
    elif entity_type == "ontology":
        uri_prefix = ONTOLOGY_URI_PREFIX
    else:
        uri_prefix = RESOURCE_URI_PREFIX

    # Wikidata: resolve labels via SPARQL (entity/France is not a valid Q-id)
    if DATASET == "wikidata" and entity_type != "property":
        resolved = _wikidata_uri_from_label(entity)
        if resolved:
            return resolved

    formatted_name = _format_entity_name(entity)
    return f"{uri_prefix}{formatted_name}"


_WIKIDATA_LABEL_CACHE = {}


def _wikidata_uri_from_label(label: str):
    """Resolve an English label to a Wikidata entity URI via wbsearchentities (cached)."""
    key = label.strip().lower()
    if key in _WIKIDATA_LABEL_CACHE:
        return _WIKIDATA_LABEL_CACHE[key]
    uri = None
    try:
        resp = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": label,
                "language": "en",
                "limit": 1,
                "format": "json",
            },
            headers={
                "User-Agent": "GSSplusplus-research/1.0 (hallucination-pilot; academic use)",
            },
            timeout=30,
        )
        resp.raise_for_status()
        hits = resp.json().get("search") or []
        if hits:
            qid = hits[0].get("id")
            if qid:
                uri = f"{RESOURCE_URI_PREFIX}{qid}"
    except Exception as e:
        print(f"    Wikidata label API error for '{label}': {e}")
    _WIKIDATA_LABEL_CACHE[key] = uri
    if uri:
        print(f"    Wikidata label '{label}' -> {uri}")
    else:
        print(f"    Wikidata label lookup miss for '{label}'")
    time.sleep(0.5)
    return uri


def is_predicate_uri(uri: str) -> bool:
    """Checks if a URI is a predicate (property or ontology)."""
    if DATASET == "wikidata":
        return uri.startswith(PROPERTY_URI_PREFIX) or uri.startswith("http://www.wikidata.org/prop/")
    else:
        return uri.startswith("http://dbpedia.org/property/") or uri.startswith("http://dbpedia.org/ontology/")


def is_resource_uri(uri: str) -> bool:
    """Checks if a URI is a resource."""
    if DATASET == "wikidata":
        return uri.startswith(RESOURCE_URI_PREFIX) or uri.startswith("http://www.wikidata.org/entity/")
    else:
        return uri.startswith("http://dbpedia.org/resource/")


def extract_entity_name_from_uri(uri: str) -> str:
    """Extracts a readable entity name from a URI."""
    if not uri:
        return None
    
    if DATASET == "wikidata":
        if uri.startswith(RESOURCE_URI_PREFIX) or uri.startswith("http://www.wikidata.org/entity/"):
            name = uri.replace(RESOURCE_URI_PREFIX, "").replace("http://www.wikidata.org/entity/", "")
            return name.replace("_", " ").lower()
        elif uri.startswith(PROPERTY_URI_PREFIX) or uri.startswith("http://www.wikidata.org/prop/"):
            name = uri.replace(PROPERTY_URI_PREFIX, "").replace("http://www.wikidata.org/prop/direct/", "").replace("http://www.wikidata.org/prop/", "")
            return name.replace("_", " ").lower()
    else:
        if uri.startswith("http://dbpedia.org/resource/"):
            return uri.replace("http://dbpedia.org/resource/", "").replace("_", " ").lower()
        elif uri.startswith("http://dbpedia.org/property/"):
            return uri.replace("http://dbpedia.org/property/", "").replace("_", " ").lower()
        elif uri.startswith("http://dbpedia.org/ontology/"):
            return uri.replace("http://dbpedia.org/ontology/", "").replace("_", " ").lower()

    return None


def _get_noisy_predicates():
    """Returns list of noisy predicates to filter based on dataset."""
    if DATASET == "wikidata":
        return [
            "rdf:type",
            "rdfs:label",
            "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
            "<http://www.w3.org/2000/01/rdf-schema#label>",
            "<http://schema.org/description>",
            "<http://www.w3.org/2000/01/rdf-schema#comment>",
            "<http://www.w3.org/2000/01/rdf-schema#seeAlso>",
        ]
    else:
        return [
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


_NOISY_PREDICATES = _get_noisy_predicates()


def _parse_triples_from_bindings(bindings: list) -> list:
    """Helper to parse triples from SPARQL bindings."""
    triples = []
    for binding in bindings:
        triple = {var: binding[var].get("value", "") for var in ["s", "p", "o"] if var in binding}
        if triple:
            triples.append(triple)
    return triples


def _get_resource_uri_filter() -> str:
    """Returns the resource URI filter string for SPARQL queries."""
    if DATASET == "wikidata":
        return f'STRSTARTS(STR(?uri), "{RESOURCE_URI_PREFIX}")'
    else:
        return 'STRSTARTS(STR(?uri), "http://dbpedia.org/resource/")'


def generate_degree_query(uri: str) -> str:
    """Generates a SPARQL query to compute both out-degree and in-degree for a URI."""
    predicate_filter = _get_predicate_filter_string()
    resource_filter = _get_resource_uri_filter().replace("?uri", "?s").replace("?uri", "?o")

    if is_predicate_uri(uri):
        if DATASET == "wikidata":
            sparql_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        
        SELECT 
            (COUNT(DISTINCT ?s) AS ?out_degree)
            (COUNT(DISTINCT ?o) AS ?in_degree)
        WHERE {{
            ?s <{uri}> ?o .
            FILTER(isIRI(?s))
            FILTER(isIRI(?o))
            FILTER(STRSTARTS(STR(?s), "{RESOURCE_URI_PREFIX}"))
            FILTER(STRSTARTS(STR(?o), "{RESOURCE_URI_PREFIX}"))
        }}
        """
        else:
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
        if DATASET == "wikidata":
            sparql_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        
        SELECT 
            (COUNT(DISTINCT ?o_out) AS ?out_degree)
            (COUNT(DISTINCT ?s_in) AS ?in_degree)
        WHERE {{
            OPTIONAL {{
                <{uri}> ?p_out ?o_out .
                FILTER(isIRI(?o_out))
                FILTER(STRSTARTS(STR(?o_out), "{RESOURCE_URI_PREFIX}"))
                FILTER(?p_out NOT IN (
                    {predicate_filter}
                ))
            }}
            OPTIONAL {{
                ?s_in ?p_in <{uri}> .
                FILTER(isIRI(?s_in))
                FILTER(STRSTARTS(STR(?s_in), "{RESOURCE_URI_PREFIX}"))
                FILTER(?p_in NOT IN (
                    {predicate_filter}
                ))
            }}
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
    """Computes weighted degree centrality for a URI."""
    result_data = execute_sparql_query_with_retry(generate_degree_query(uri), max_retries=2, initial_timeout=45)

    out_degree = in_degree = 0
    if result_data and "results" in result_data and "bindings" in result_data["results"]:
        binding = result_data["results"]["bindings"][0] if result_data["results"]["bindings"] else {}
        out_degree = int(binding.get("out_degree", {}).get("value", "0"))
        in_degree = int(binding.get("in_degree", {}).get("value", "0"))

    return (out_degree, in_degree, out_degree + in_degree)


def _get_predicate_filter_string() -> str:
    """Returns formatted predicate filter string."""
    return ",\n                ".join(_NOISY_PREDICATES)


def _build_resource_pattern(resource: str, as_subject: bool = True, predicate: str = None) -> str:
    """Builds a SPARQL pattern for a resource."""
    resource_filter_prefix = RESOURCE_URI_PREFIX if DATASET == "wikidata" else "http://dbpedia.org/resource/"
    
    if as_subject:
        if predicate:
            return f"""
                {{
                    <{resource}> <{predicate}> ?o .
                    BIND(<{resource}> AS ?s)
                    BIND(<{predicate}> AS ?p)
                    FILTER(isIRI(?o))
                    FILTER(STRSTARTS(STR(?o), "{resource_filter_prefix}"))
                }}"""
        else:
            predicate_filter = _get_predicate_filter_string()
            return f"""
            {{
                <{resource}> ?p ?o .
                BIND(<{resource}> AS ?s)
                FILTER(isIRI(?o))
                FILTER(STRSTARTS(STR(?o), "{resource_filter_prefix}"))
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
                    FILTER(STRSTARTS(STR(?s), "{resource_filter_prefix}"))
                }}"""
        else:
            predicate_filter = _get_predicate_filter_string()
            return f"""
            {{
                ?s ?p <{resource}> .
                BIND(<{resource}> AS ?o)
                FILTER(isIRI(?s))
                FILTER(STRSTARTS(STR(?s), "{resource_filter_prefix}"))
                FILTER(?p NOT IN (
                    {predicate_filter}
                ))
            }}"""


def _build_predicate_pattern(predicate: str) -> str:
    """Builds a SPARQL pattern for a predicate."""
    resource_filter_prefix = RESOURCE_URI_PREFIX if DATASET == "wikidata" else "http://dbpedia.org/resource/"
    return f"""
            {{
                ?s <{predicate}> ?o .
                BIND(<{predicate}> AS ?p)
                FILTER(isIRI(?s))
                FILTER(isIRI(?o))
                FILTER(STRSTARTS(STR(?s), "{resource_filter_prefix}"))
                FILTER(STRSTARTS(STR(?o), "{resource_filter_prefix}"))
            }}"""


def generate_batch_retrieve_query(resource_uris: list, predicate_uris: list, limit: int = None) -> str:
    """Generates a SPARQL query to retrieve triples using resources and/or predicates."""
    if limit is None:
        limit = KG_DEFAULT_TRIPLE_LIMIT

    if not resource_uris and not predicate_uris:
        return None

    union_patterns = []

    if resource_uris and predicate_uris:
        max_resources = min(len(resource_uris), 10)
        max_predicates = min(len(predicate_uris), 5)

        for resource in resource_uris[:max_resources]:
            for predicate in predicate_uris[:max_predicates]:
                if len(union_patterns) >= KG_MAX_UNION_CLAUSES:
                    break
                union_patterns.append(_build_resource_pattern(resource, as_subject=True, predicate=predicate))
                if len(union_patterns) >= KG_MAX_UNION_CLAUSES:
                    break
                union_patterns.append(_build_resource_pattern(resource, as_subject=False, predicate=predicate))
            if len(union_patterns) >= KG_MAX_UNION_CLAUSES:
                break

        if len(union_patterns) < KG_MAX_UNION_CLAUSES:
            remaining = KG_MAX_UNION_CLAUSES - len(union_patterns)
            for resource in resource_uris[:min(len(resource_uris), remaining // 2)]:
                if len(union_patterns) >= KG_MAX_UNION_CLAUSES:
                    break
                union_patterns.append(_build_resource_pattern(resource, as_subject=True))
                if len(union_patterns) >= KG_MAX_UNION_CLAUSES:
                    break
                union_patterns.append(_build_resource_pattern(resource, as_subject=False))
    elif resource_uris:
        max_resources = min(len(resource_uris), KG_MAX_UNION_CLAUSES // 2)
        for resource in resource_uris[:max_resources]:
            union_patterns.append(_build_resource_pattern(resource, as_subject=True))
            union_patterns.append(_build_resource_pattern(resource, as_subject=False))
    else:
        max_predicates = min(len(predicate_uris), KG_MAX_UNION_CLAUSES)
        for predicate in predicate_uris[:max_predicates]:
            union_patterns.append(_build_predicate_pattern(predicate))

    if not union_patterns:
        return None

    union_clauses = "\n            UNION".join(union_patterns)

    if DATASET == "wikidata":
        return f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT ?s ?p ?o
    WHERE {{
        {union_clauses}
    }}
    LIMIT {limit}
    """
    else:
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
    """Fetches triples in batch where subject/object is any resource or predicate is any predicate."""
    if limit is None:
        limit = KG_DEFAULT_TRIPLE_LIMIT

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
    """Extracts all URIs (resources, properties, ontologies) from triples."""
    all_uris_in_triples = set()
    for triple in all_triples:
        for key in ["s", "o", "p"]:
            uri = triple.get(key, "")
            if DATASET == "wikidata":
                if uri.startswith("http://www.wikidata.org/"):
                    all_uris_in_triples.add(uri)
            else:
                if uri.startswith("http://dbpedia.org/"):
                    all_uris_in_triples.add(uri)
    return all_uris_in_triples
