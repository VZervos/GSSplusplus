"""
Generic LLM client that supports multiple providers (Gemini, OpenAI, Anthropic).
"""
import requests
from typing import Dict, Any, Optional

from config.settings import (
    LLM_PROVIDER,
    LLM_API_KEY,
    LLM_MODEL_NAME,
    LLM_API_URL,
    LLM_BASE_URL
)


def call_llm_api(prompt: str, api_key: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Call the configured LLM API with the given prompt.
    
    Args:
        prompt: The prompt text to send to the LLM
        api_key: Optional API key override (defaults to configured key)
        timeout: Request timeout in seconds (default: 60)
    
    Returns:
        Dictionary containing the API response
    
    Raises:
        Exception: If the API request fails
    """
    if api_key is None:
        api_key = LLM_API_KEY
    
    if LLM_PROVIDER == "openai":
        return _call_openai_api(prompt, api_key, timeout)
    elif LLM_PROVIDER == "anthropic":
        return _call_anthropic_api(prompt, api_key, timeout)
    else:  # Default to Gemini
        return _call_gemini_api(prompt, api_key, timeout)


def _call_gemini_api(prompt: str, api_key: str, timeout: int) -> Dict[str, Any]:
    """Call Google Gemini API."""
    url = f"{LLM_API_URL}?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    
    print(f"Sending request to Gemini API (model: {LLM_MODEL_NAME})...")
    print(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else prompt)
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            error_json = response.json() if response.content else {}
            error_msg = error_json.get("error", {}).get("message", f"HTTP {response.status_code}")
            raise Exception(f"Gemini API Error: {error_msg}")
        
        return response.json()
    
    except requests.exceptions.Timeout:
        raise Exception(f"Gemini API request timeout after {timeout} seconds")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Gemini API network error: {str(e)}")


def _call_openai_api(prompt: str, api_key: str, timeout: int) -> Dict[str, Any]:
    """Call OpenAI API."""
    url = LLM_API_URL
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    print(f"Sending request to OpenAI API (model: {LLM_MODEL_NAME})...")
    print(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else prompt)
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            error_json = response.json() if response.content else {}
            error_msg = error_json.get("error", {}).get("message", f"HTTP {response.status_code}")
            raise Exception(f"OpenAI API Error: {error_msg}")
        
        return response.json()
    
    except requests.exceptions.Timeout:
        raise Exception(f"OpenAI API request timeout after {timeout} seconds")
    except requests.exceptions.RequestException as e:
        raise Exception(f"OpenAI API network error: {str(e)}")


def _call_anthropic_api(prompt: str, api_key: str, timeout: int) -> Dict[str, Any]:
    """Call Anthropic (Claude) API."""
    url = LLM_API_URL
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": LLM_MODEL_NAME,
        "max_tokens": 2000,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    print(f"Sending request to Anthropic API (model: {LLM_MODEL_NAME})...")
    print(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else prompt)
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            error_json = response.json() if response.content else {}
            error_msg = error_json.get("error", {}).get("message", f"HTTP {response.status_code}")
            raise Exception(f"Anthropic API Error: {error_msg}")
        
        return response.json()
    
    except requests.exceptions.Timeout:
        raise Exception(f"Anthropic API request timeout after {timeout} seconds")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Anthropic API network error: {str(e)}")


def extract_response_text(api_response: Dict[str, Any]) -> str:
    """
    Extract text from LLM API response based on the configured provider.
    
    Args:
        api_response: The API response dictionary
    
    Returns:
        Extracted text content
    """
    if LLM_PROVIDER == "openai":
        try:
            return api_response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            return str(api_response)
    elif LLM_PROVIDER == "anthropic":
        try:
            return api_response["content"][0]["text"]
        except (KeyError, IndexError) as e:
            return str(api_response)
    else:  # Gemini
        try:
            return api_response["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            return str(api_response)

