import json

def extract_response_text(api_response: dict) -> str:
    try:
        return api_response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        # If structure is different, return formatted JSON
        return json.dumps(api_response, indent=2)