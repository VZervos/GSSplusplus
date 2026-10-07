"""Local Ollama chat helper for the CoLoTa demo."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

TF_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["TRUE", "FALSE"]},
    },
    "required": ["answer"],
}

TF_OR_INSUFFICIENT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "enum": ["TRUE", "FALSE", "INSUFFICIENT"],
        },
    },
    "required": ["answer"],
}


def parse_tf(text: str, allow_insufficient: bool = False) -> str:
    raw = (text or "").strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "answer" in obj:
            raw = str(obj["answer"])
        elif isinstance(obj, bool):
            return "True" if obj else "False"
    except json.JSONDecodeError:
        pass

    upper = raw.upper()
    for ch in ".:;!,\"'{}":
        upper = upper.replace(ch, " ")
    tokens = upper.split()
    first = tokens[0] if tokens else ""

    if allow_insufficient and first == "INSUFFICIENT":
        return "INSUFFICIENT"
    if first == "TRUE":
        return "True"
    if first == "FALSE":
        return "False"

    if allow_insufficient and "INSUFFICIENT" in upper:
        return "INSUFFICIENT"

    true_i = upper.rfind("TRUE")
    false_i = upper.rfind("FALSE")
    if true_i == -1 and false_i == -1:
        return "UNKNOWN"
    return "True" if true_i > false_i else "False"


def chat(
    prompt: str,
    model: str | None = None,
    timeout: int = 300,
    schema: dict | None = None,
    system: str | None = None,
) -> str:
    model = model or OLLAMA_MODEL
    url = f"{OLLAMA_BASE_URL}/api/chat"
    if system is None:
        if schema is TF_OR_INSUFFICIENT_SCHEMA:
            system = (
                "You answer yes/no questions from provided knowledge. "
                'Respond with JSON only: {"answer": "TRUE"}, '
                '{"answer": "FALSE"}, or {"answer": "INSUFFICIENT"}.'
            )
        else:
            system = (
                "You answer yes/no questions. "
                'Respond with JSON only: {"answer": "TRUE"} or '
                '{"answer": "FALSE"}.'
            )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": schema or TF_SCHEMA,
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
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama request failed ({url}, model={model}): {exc}"
        ) from exc

    text = (data.get("message") or {}).get("content", "")
    if not text:
        raise RuntimeError(f"Ollama returned an empty response: {data}")
    return text
