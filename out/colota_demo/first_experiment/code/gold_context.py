"""Condition 2 — gold KG triples (oracle evidence, no inference rule)."""

from __future__ import annotations

from llm_client import TF_SCHEMA, chat, parse_tf


def format_triples(triples: list[str]) -> str:
    return "\n".join(f"- {triple}" for triple in triples)


def build_prompt(question: str, triples: list[str]) -> str:
    context = format_triples(triples)
    return f"""Question:
{question}

Relevant knowledge from a knowledge graph:
{context}

Use this knowledge to decide. It is enough to answer TRUE or FALSE.
Do not use outside facts.

Return JSON: {{"answer": "TRUE"}} or {{"answer": "FALSE"}}.
"""


def predict(
    question: str,
    triples: list[str],
    model: str | None = None,
) -> tuple[str, str]:
    raw = chat(build_prompt(question, triples), model=model, schema=TF_SCHEMA)
    return parse_tf(raw, allow_insufficient=False), raw
