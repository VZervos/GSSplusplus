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

def retrieve_candidate_triples(uri, limit=100):
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

def fetch_triples_from_sparql(sparql_query):
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