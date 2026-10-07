"""Condition 1 — LLM only (parametric knowledge, no KG context)."""

from __future__ import annotations

from llm_client import TF_SCHEMA, chat, parse_tf


def build_prompt(question: str) -> str:
    return f"""Question:
{question}

Decide if the answer is TRUE or FALSE.
Return JSON: {{"answer": "TRUE"}} or {{"answer": "FALSE"}}.
"""


def predict(question: str, model: str | None = None) -> tuple[str, str]:
    raw = chat(build_prompt(question), model=model, schema=TF_SCHEMA)
    return parse_tf(raw, allow_insufficient=False), raw
