"""Oracle subgraph: gold CoLoTa triples only, no inference rule."""

from __future__ import annotations

from utils.dataset import item_triples
from utils.llm_client import TF_SCHEMA, chat, parse_tf

PROMPT = """You are given a yes/no question and a small set of knowledge-graph
triples that are sufficient to answer it. Use only these triples.

Question: {question}

Triples:
{triples}

Return JSON: {{"answer": "TRUE"}} or {{"answer": "FALSE"}}.
"""


class GoldTriplesApproach:
    name = "gold_triples"
    needs_corpus = False

    def prepare(self, corpus: list[dict], config: dict) -> None:
        return None

    def predict(self, question: str, *, item: dict, model: str, **_kwargs) -> str:
        triples = item_triples(item)
        prompt = PROMPT.format(
            question=question,
            triples="\n".join(f"- {t}" for t in triples) or "(none)",
        )
        raw = chat(prompt, model=model, schema=TF_SCHEMA)
        return parse_tf(raw)
