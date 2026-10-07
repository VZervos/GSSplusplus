import os, sys, requests, json



def get_wikidata_types(object_name):
    # Step 1: see if the object name exists in geo_data.json

    with open('data/geo_data.json', 'r') as f:
        geo_data = json.load(f)

    with open('data/onto_data.json', 'r') as f:
        onto_data = json.load(f)
    

    with open('data/term2types.json', 'r') as f:    
        term2types = json.load(f)
    
    if object_name in term2types.keys():
        return term2types[object_name]
    else:
        print(object_name)
    for entry in onto_data:
        if object_name in entry['types'].keys():
            return entry['types'][object_name]

    for entry in geo_data:
        if object_name in entry['types'].keys():
            return entry['types'][object_name]
        



    search_url = 'https://www.wikidata.org/w/api.php'
    search_params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'search': object_name
    }
    search_response = requests.get(search_url, params=search_params).json()
    
    if not search_response['search']:
        return None
    
    wikidata_id = search_response['search'][0]['id']

    # Step 2: Get the types for the Wikidata ID
    query_url = f'https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json'
    query_response = requests.get(query_url).json()
    
    if 'entities' not in query_response or wikidata_id not in query_response['entities']:
        return None
    
    entity_data = query_response['entities'][wikidata_id]
    claims = entity_data['claims']
    
    types = set()

    # Extract P31 (instance of) and P279 (subclass of) types
    if 'P31' in claims:
        for claim in claims['P31']:
            if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                types.add(claim['mainsnak']['datavalue']['value']['id'])

    if 'P279' in claims:
        for claim in claims['P279']:
            if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                types.add(claim['mainsnak']['datavalue']['value']['id'])

    types_list = list(types)

    types_list_labels = [get_label(qid) for qid in types_list]

    return types_list_labels



def get_label(qid):
    url = f"https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbgetentities',
        'ids': qid,
        'format': 'json',
        'props': 'labels',
        'languages': 'en'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    try:
        label = data['entities'][qid]['labels']['en']['value']
        return label
    except KeyError:
        return None