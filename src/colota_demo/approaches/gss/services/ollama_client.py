"""Local Ollama LLM client (Gemini-compatible response shape)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL


def call_ollama_api(
    prompt: str,
    model: str = None,
    base_url: str = None,
    timeout: int = 300,
) -> dict:
    model = model or OLLAMA_MODEL
    base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
    url = f"{base_url}/api/chat"

    print(f"Sending request to Ollama (model: {model})...")
    print(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else prompt)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise Exception(f"Ollama API error: {e}") from e

    response_text = (data.get("message") or {}).get("content", "")
    if not response_text:
        raise Exception(f"Ollama returned empty response: {data}")

    print("Response received from Ollama")
    return {
        "candidates": [{
            "content": {
                "parts": [{"text": response_text}]
            }
        }]
    }
