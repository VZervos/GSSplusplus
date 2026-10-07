"""Parametric knowledge baseline: no retrieved or gold context."""

from __future__ import annotations

from utils.llm_client import TF_SCHEMA, chat, parse_tf

PROMPT = """Answer the following yes/no question using only your own knowledge.

Question: {question}

Return JSON: {{"answer": "TRUE"}} or {{"answer": "FALSE"}}.
"""


class LlmOnlyApproach:
    name = "llm_only"
    needs_corpus = False

    def prepare(self, corpus: list[dict], config: dict) -> None:
        return None

    def predict(self, question: str, *, model: str, **_kwargs) -> str:
        raw = chat(PROMPT.format(question=question), model=model, schema=TF_SCHEMA)
        return parse_tf(raw)
