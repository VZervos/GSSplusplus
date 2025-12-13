import urllib
import requests


def lookup_entity_uri(entity):
    """
    Converts an entity name to a DBpedia URI.
    Formats the entity name by capitalizing words and replacing spaces with underscores.
    """
    # Format entity name for DBpedia: capitalize each word and replace spaces with underscores
    formatted_entity = entity.replace(" ", "_")
    # Capitalize first letter of each word (DBpedia convention)
    parts = formatted_entity.split("_")
    formatted_entity = "_".join([part.capitalize() if part else "" for part in parts])
    
    return (
        f"http://dbpedia.org/resource/{formatted_entity}"
    )

def generate_retrieve_query(uri, limit=100):
    sparql_query = f"""
    SELECT ?s ?p ?o
    WHERE {{
        {{
            <{uri}> ?p ?o .
            BIND(<{uri}> AS ?s)
        }}
        UNION
        {{
            ?s <{uri}> ?o .
            BIND(<{uri}> AS ?p)
        }}
        UNION
        {{
            ?s ?p <{uri}> .
            BIND(<{uri}> AS ?o)
        }}
    }}
    LIMIT {limit}
    """
    return sparql_query

def fetch_triples_from_sparql(uri):
    sparql_query = generate_retrieve_query(uri)
    """
    Fetches triples from DBpedia SPARQL endpoint using POST request and returns the results.
    """
    try:
        sparql_endpoint = "https://dbpedia.org/sparql"
        headers = {
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Mozilla/5.0"
        }
        
        # Send query as POST data
        data = {
            "query": sparql_query,
            "format": "json"
        }
        
        response = requests.post(sparql_endpoint, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        
        # Parse JSON response
        result_data = response.json()
        
        # Extract triples from SPARQL JSON response
        triples = []
        if "results" in result_data and "bindings" in result_data["results"]:
            for binding in result_data["results"]["bindings"]:
                triple = {}
                for var in ["s", "p", "o"]:
                    if var in binding:
                        triple[var] = binding[var].get("value", "")
                if triple:
                    triples.append(triple)
        return triples
    except requests.exceptions.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON response. Response text: {response.text[:500]}")
        return []
    except Exception as e:
        print(f"ERROR fetching triples: {str(e)}")
        if 'response' in locals():
            print(f"Response status: {response.status_code}")
            print(f"Response preview: {response.text[:200]}")
        return []


def get_pagerank(uri):
    """
    Attempts to retrieve PageRank score for a URI from DBpedia.
    Returns the PageRank value if available, None otherwise.
    """
    try:
        sparql_query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        SELECT ?rank
        WHERE {{
            <{uri}> dbo:wikiPageRank ?rank .
        }}
        LIMIT 1
        """
        
        sparql_endpoint = "https://dbpedia.org/sparql"
        headers = {
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Mozilla/5.0"
        }
        
        data = {
            "query": sparql_query,
            "format": "json"
        }
        
        response = requests.post(sparql_endpoint, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        result_data = response.json()
        
        if "results" in result_data and "bindings" in result_data["results"]:
            bindings = result_data["results"]["bindings"]
            if bindings and "rank" in bindings[0]:
                rank_value = bindings[0]["rank"].get("value", "")
                try:
                    return float(rank_value)
                except (ValueError, TypeError):
                    return None
        return None
    except Exception:
        return None


def compute_weighted_degree(uri):
    """
    Computes weighted degree centrality for a URI.
    Counts entity-to-entity links (excluding noisy predicates like rdf:type, rdfs:label).
    Returns a tuple: (out_degree, in_degree, total_degree)
    """
    try:
        # Query for out-degree (URI as subject, linking to other entities)
        out_degree_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT (COUNT(DISTINCT ?o) AS ?count)
        WHERE {{
            <{uri}> ?p ?o .
            FILTER(isIRI(?o))
            FILTER(?p NOT IN (rdf:type, rdfs:label, <http://dbpedia.org/ontology/wikiPageID>))
        }}
        """
        
        # Query for in-degree (URI as object, linked from other entities)
        in_degree_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT (COUNT(DISTINCT ?s) AS ?count)
        WHERE {{
            ?s ?p <{uri}> .
            FILTER(isIRI(?s))
            FILTER(?p NOT IN (rdf:type, rdfs:label, <http://dbpedia.org/ontology/wikiPageID>))
        }}
        """
        
        sparql_endpoint = "https://dbpedia.org/sparql"
        headers = {
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Mozilla/5.0"
        }
        
        # Get out-degree
        out_degree = 0
        try:
            data = {"query": out_degree_query, "format": "json"}
            response = requests.post(sparql_endpoint, headers=headers, data=data, timeout=30)
            response.raise_for_status()
            result_data = response.json()
            if "results" in result_data and "bindings" in result_data["results"]:
                bindings = result_data["results"]["bindings"]
                if bindings and "count" in bindings[0]:
                    out_degree = int(bindings[0]["count"].get("value", "0"))
        except Exception:
            pass
        
        # Get in-degree
        in_degree = 0
        try:
            data = {"query": in_degree_query, "format": "json"}
            response = requests.post(sparql_endpoint, headers=headers, data=data, timeout=30)
            response.raise_for_status()
            result_data = response.json()
            if "results" in result_data and "bindings" in result_data["results"]:
                bindings = result_data["results"]["bindings"]
                if bindings and "count" in bindings[0]:
                    in_degree = int(bindings[0]["count"].get("value", "0"))
        except Exception:
            pass
        
        total_degree = out_degree + in_degree
        return (out_degree, in_degree, total_degree)
        
    except Exception as e:
        print(f"ERROR computing weighted degree for {uri}: {str(e)}")
        return (0, 0, 0)


def compute_importance(uri):
    """
    Computes importance score for a URI.
    First tries PageRank, falls back to weighted degree centrality.
    Returns a dictionary with importance metrics.
    """
    # Try PageRank first
    pagerank = get_pagerank(uri)
    
    if pagerank is not None:
        return {
            "method": "pagerank",
            "score": pagerank,
            "out_degree": None,
            "in_degree": None,
            "total_degree": None
        }
    
    # Fallback to weighted degree
    out_degree, in_degree, total_degree = compute_weighted_degree(uri)
    
    # Normalize the degree score (using log scale to handle large differences)
    # This gives a score between 0 and ~10 for most entities
    normalized_score = 0.0
    if total_degree > 0:
        normalized_score = min(10.0, 1.0 + (total_degree ** 0.5) / 10.0)
    
    return {
        "method": "weighted_degree",
        "score": normalized_score,
        "out_degree": out_degree,
        "in_degree": in_degree,
        "total_degree": total_degree
    }