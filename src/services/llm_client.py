from config.settings import (
    LLM_PROVIDER,
    API_KEY,
    GROQ_API_KEY,
)
from services.gemini_client import call_gemini_api
from services.groq_client import call_groq_api
from services.ollama_client import call_ollama_api


def _has_real_key(value: str) -> bool:
    return bool(value) and value.strip() not in {"", "YOUR KEY"}


def _resolve_provider() -> str:
    provider = (LLM_PROVIDER or "auto").lower().strip()
    if provider == "auto":
        if _has_real_key(GROQ_API_KEY):
            return "groq"
        return "ollama"
    return provider


def call_llm_api(prompt: str, timeout: int = 300) -> dict:
    provider = _resolve_provider()

    if provider == "gemini":
        print("Using LLM provider: Gemini")
        return call_gemini_api(prompt, api_key=API_KEY, timeout=timeout)

    if provider == "groq":
        print("Using LLM provider: Groq")
        try:
            return call_groq_api(prompt, api_key=GROQ_API_KEY, timeout=timeout)
        except Exception as e:
            if "invalid_api_key" in str(e).lower() or "authentication" in str(e).lower():
                print(f"Groq failed ({e}); falling back to Ollama")
                print("Using LLM provider: Ollama")
                return call_ollama_api(prompt, timeout=timeout)
            raise

    if provider == "ollama":
        print("Using LLM provider: Ollama")
        return call_ollama_api(prompt, timeout=timeout)

    raise ValueError(
        f"Unknown LLM_PROVIDER: '{LLM_PROVIDER}'. "
        f"Supported providers: 'gemini', 'groq', 'ollama', 'auto'."
    )


def extract_response_text(response: dict) -> str:
    """Extract plain text from a Gemini-compatible LLM response dict."""
    try:
        return response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected LLM response shape: {response}") from e
