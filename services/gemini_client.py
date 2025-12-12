import requests
import json
from config.settings import API_KEY, API_URL

def call_gemini_api(prompt: str, api_key: str = API_KEY, timeout: int = 30):
    url = f"{API_URL}?key={api_key}"

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    print(f"📤 Sending request to Gemini API...")
    print(f"📝 Prompt: {prompt[:50]}..." if len(prompt) > 50 else prompt)

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        print(f"✅ Response status: {response.status_code}")

        if response.status_code != 200:
            error_json = response.json() if response.content else {}
            error_msg = error_json.get("error", {}).get("message", f"HTTP {response.status_code}")
            raise Exception(f"API Error: {error_msg}")

        return response.json()

    except requests.exceptions.Timeout:
        raise Exception(f"⏱️ Request timeout after {timeout} seconds")

    except requests.exceptions.RequestException as e:
        raise Exception(f"❌ Network error: {str(e)}")