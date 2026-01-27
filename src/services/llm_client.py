from config.settings import LLM_PROVIDER, API_KEY, GROQ_API_KEY
from services.gemini_client import call_gemini_api
from services.groq_client import call_groq_api

def call_llm_api(prompt: str, timeout: int = 30) -> dict:
    provider = LLM_PROVIDER.lower().strip()
    
    if provider == "gemini":
        print(f"Using LLM provider: Gemini")
        return call_gemini_api(prompt, api_key=API_KEY, timeout=timeout)
    
    elif provider == "groq":
        print(f"Using LLM provider: Groq")
        return call_groq_api(prompt, api_key=GROQ_API_KEY, timeout=timeout)
    
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. "
            f"Supported providers: 'gemini', 'groq'. "
            f"Please set the LLM_PROVIDER environment variable to a valid provider."
        )

