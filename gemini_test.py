import requests
import json
import sys
from typing import Optional

# Your API key
API_KEY = "AIzaSyBf7J4568JtOP122Q820BO4D05AfYLdJ6A"

# API endpoint for Gemini 2.5 Flash change model name to use different models
model_name = "gemini-2.5-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"


def call_gemini_api(prompt: str, api_key: str = API_KEY, timeout: int = 30) -> dict:
    url = f"{API_URL}?key={api_key}"
    
    headers = {
        "Content-Type": "application/json",
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    print(f"📤 Sending request to Gemini API...")
    print(f"📝 Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"📝 Prompt: {prompt}")
    print()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        
        print(f"✅ Response status: {response.status_code}")
        
        if response.status_code != 200:
            error_data = response.json() if response.content else {}
            error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
            raise Exception(f"API Error: {error_msg}")
        
        return response.json()
    
    except requests.exceptions.Timeout:
        raise Exception(f"⏱️ Request timeout after {timeout} seconds")
    except requests.exceptions.RequestException as e:
        raise Exception(f"❌ Network error: {str(e)}")


def extract_response_text(api_response: dict) -> str:
    try:
        return api_response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        # If structure is different, return formatted JSON
        return json.dumps(api_response, indent=2)


def main():
    """Main function to run the script."""
    # Get prompt from command line argument or use default
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        print("💡 No prompt provided, using default prompt.")
        print()
        sys.exit(1)
    
    try:
        response = call_gemini_api(prompt)
        response_text = extract_response_text(response)
        
        print("=" * 60)
        print("🤖 GEMINI RESPONSE:")
        print("=" * 60)
        print(response_text)
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print("❌ ERROR:")
        print("=" * 60)
        print(str(e))
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

