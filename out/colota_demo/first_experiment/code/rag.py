"""Condition 3 — RAG over the pooled gold-triple corpus."""

from __future__ import annotations

from llm_client import TF_OR_INSUFFICIENT_SCHEMA, chat, parse_tf


def build_prompt(question: str, retrieved: list[dict]) -> str:
    context = "\n".join(
        f"- {item['text']}"
        for item in retrieved
    )
    return f"""Question:
{question}

Retrieved knowledge:
{context}

Use only the retrieved knowledge.
If it is not enough to decide, return {{"answer": "INSUFFICIENT"}}.
Otherwise return {{"answer": "TRUE"}} or {{"answer": "FALSE"}}.
"""


def predict(
    question: str,
    retrieved: list[dict],
    model: str | None = None,
) -> tuple[str, str]:
    raw = chat(
        build_prompt(question, retrieved),
        model=model,
        schema=TF_OR_INSUFFICIENT_SCHEMA,
    )
    return parse_tf(raw, allow_insufficient=True), raw
