import requests
import time


def execute_sparql_query_with_retry(sparql_query, max_retries=2, initial_timeout=45, max_timeout=90):
    """
    Executes a SPARQL query with retry logic and exponential backoff.
    Only retries on timeouts (transient issues), not on other errors.
    
    Args:
        sparql_query: The SPARQL query string
        max_retries: Maximum number of retry attempts (default: 2)
        initial_timeout: Initial timeout in seconds (default: 45)
        max_timeout: Maximum timeout in seconds (default: 90)
    
    Returns:
        dict: JSON response from SPARQL endpoint, or None if all retries fail
    """
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
                
        except requests.exceptions.RequestException as e:
            print(f"    Error: {str(e)}")
            return None
    
    return None


def lookup_entity_uri(entity):
    """Converts an entity name to a DBpedia URI."""
    formatted_entity = "_".join(part.capitalize() for part in entity.replace(" ", "_").split("_") if part)
    return f"http://dbpedia.org/resource/{formatted_entity}"


def extract_entity_name_from_uri(uri):
    """Extracts a readable entity name from a DBpedia URI."""
    if not uri or not uri.startswith("http://dbpedia.org/resource/"):
        return None
    return uri.replace("http://dbpedia.org/resource/", "").replace("_", " ").lower()


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

def generate_retrieve_query(uri, limit=100):
    """Generates a SPARQL query to retrieve triples for a URI, excluding noisy predicates."""
    predicate_filter = ",\n                ".join(_NOISY_PREDICATES)
    sparql_query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT ?s ?p ?o
    WHERE {{
        {{
            <{uri}> ?p ?o .
            BIND(<{uri}> AS ?s)
            FILTER(isIRI(?o))
            FILTER(STRSTARTS(STR(?o), "http://dbpedia.org/resource/"))
            FILTER(?p NOT IN (
                {predicate_filter}
            ))
        }}
        UNION
        {{
            ?s ?p <{uri}> .
            BIND(<{uri}> AS ?o)
            FILTER(isIRI(?s))
            FILTER(STRSTARTS(STR(?s), "http://dbpedia.org/resource/"))
            FILTER(?p NOT IN (
                {predicate_filter}
            ))
        }}
    }}
    LIMIT {limit}
    """
    return sparql_query


def _parse_triples_from_bindings(bindings):
    """Helper to parse triples from SPARQL bindings."""
    triples = []
    for binding in bindings:
        triple = {var: binding[var].get("value", "") for var in ["s", "p", "o"] if var in binding}
        if triple:
            triples.append(triple)
    return triples

def fetch_triples_with_importance(uri, limit=50):
    """Fetches triples and computes importance (in/out degree) for a URI."""
    triples = []
    out_degree = 0
    in_degree = 0
    
    result_data = execute_sparql_query_with_retry(generate_retrieve_query(uri, limit), max_retries=2, initial_timeout=45)
    if result_data and "results" in result_data and "bindings" in result_data["results"]:
        triples = _parse_triples_from_bindings(result_data["results"]["bindings"])
    
    result_data = execute_sparql_query_with_retry(generate_degree_query(uri), max_retries=2, initial_timeout=45)
    if result_data and "results" in result_data and "bindings" in result_data["results"]:
        bindings = result_data["results"]["bindings"]
        if bindings:
            binding = bindings[0]
            out_degree = int(binding.get("out_degree", {}).get("value", "0"))
            in_degree = int(binding.get("in_degree", {}).get("value", "0"))
    
    return {
        "triples": triples,
        "out_degree": out_degree,
        "in_degree": in_degree,
        "total_degree": out_degree + in_degree
    }


def generate_degree_query(uri):
    """Generates a SPARQL query to compute both out-degree and in-degree for a URI."""
    predicate_filter = ",\n                ".join(_NOISY_PREDICATES)
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

def fetch_triples_from_sparql(uri, include_degree=False):
    """Fetches triples from DBpedia SPARQL endpoint. Optionally includes degree information."""
    result_data = execute_sparql_query_with_retry(generate_retrieve_query(uri), max_retries=2, initial_timeout=45)
    
    triples = []
    if result_data and "results" in result_data and "bindings" in result_data["results"]:
        triples = _parse_triples_from_bindings(result_data["results"]["bindings"])
    
    if include_degree:
        out_degree, in_degree, total_degree = compute_weighted_degree(uri)
        return {
            "triples": triples,
            "degree": {"out_degree": out_degree, "in_degree": in_degree, "total_degree": total_degree}
        }
    return triples


def compute_weighted_degree(uri):
    """Computes weighted degree centrality for a URI. Returns (out_degree, in_degree, total_degree)."""
    result_data = execute_sparql_query_with_retry(generate_degree_query(uri), max_retries=2, initial_timeout=45)
    
    out_degree = in_degree = 0
    if result_data and "results" in result_data and "bindings" in result_data["results"]:
        binding = result_data["results"]["bindings"][0] if result_data["results"]["bindings"] else {}
        out_degree = int(binding.get("out_degree", {}).get("value", "0"))
        in_degree = int(binding.get("in_degree", {}).get("value", "0"))
    
    return (out_degree, in_degree, out_degree + in_degree)


def get_uris_from_triples(all_triples):
    all_uris_in_triples = set()
    for triple in all_triples:
        for key in ["s", "o"]:
            uri = triple.get(key, "")
            if uri.startswith("http://dbpedia.org/resource/"):
                all_uris_in_triples.add(uri)
