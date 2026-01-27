"""
Backward compatibility wrapper for Gemini API.
This module now uses the generic LLM client.
"""
from services.llm_client import call_llm_api as _call_llm_api
from config.settings import API_KEY


def call_gemini_api(prompt: str, api_key: str = API_KEY, timeout: int = 30) -> dict:
    """
    Backward compatibility function for Gemini API calls.
    Now uses the generic LLM client which supports multiple providers.
    
    Args:
        prompt: The prompt text to send to the LLM
        api_key: Optional API key override
        timeout: Request timeout in seconds
    
    Returns:
        Dictionary containing the API response
    """
    return _call_llm_api(prompt, api_key=api_key, timeout=timeout)
