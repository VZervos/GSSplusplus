"""GSS NER/LLM routing: always local Ollama for the CoLoTa suite."""

from .ollama_client import call_ollama_api


def call_llm_api(prompt: str, timeout: int = 300) -> dict:
    print("Using LLM provider: Ollama")
    return call_ollama_api(prompt, timeout=timeout)


def extract_response_text(response: dict) -> str:
    try:
        return response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected LLM response shape: {response}") from e
