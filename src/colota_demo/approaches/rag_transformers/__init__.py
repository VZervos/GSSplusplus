"""Retrieve-then-generate with MiniLM cosine retrieval over pooled triples."""

from __future__ import annotations

from .retriever import DEFAULT_EMBED_MODEL, TripleRetriever, ranking_metrics
from utils.llm_client import TF_OR_INSUFFICIENT_SCHEMA, chat, parse_tf

PROMPT = """You are given a yes/no question and retrieved knowledge-graph triples.
The triples may be incomplete or include distractors. Use only the triples;
if they are not enough to decide, answer INSUFFICIENT.

Question: {question}

Retrieved triples:
{triples}

Return JSON: {{"answer": "TRUE"}}, {{"answer": "FALSE"}}, or {{"answer": "INSUFFICIENT"}}.
"""


class RagTransformersApproach:
    name = "rag_transformers"
    needs_corpus = True

    def __init__(self) -> None:
        self.retriever: TripleRetriever | None = None

    def prepare(self, corpus: list[dict], config: dict) -> None:
        embed_model = config.get("embed_model") or DEFAULT_EMBED_MODEL
        self.retriever = TripleRetriever(corpus, model_name=embed_model)

    def rank(self, question: str) -> list[dict]:
        if self.retriever is None:
            raise RuntimeError("rag_transformers.prepare() was not called")
        return self.retriever.retrieve(question)

    def evaluate_ranking(
        self,
        gold_triples: list[str],
        ranking: list[dict],
        ks: list[int] | tuple[int, ...],
    ) -> dict:
        return ranking_metrics(gold_triples, ranking, ks=tuple(ks))

    def predict(
        self,
        question: str,
        *,
        model: str,
        k: int,
        ranking: list[dict] | None = None,
        **_kwargs,
    ) -> str:
        hits = (ranking or self.rank(question))[:k]
        prompt = PROMPT.format(
            question=question,
            triples="\n".join(
                f"- {h['triple']}  (score={h['score']:.3f})" for h in hits
            )
            or "(none)",
        )
        raw = chat(prompt, model=model, schema=TF_OR_INSUFFICIENT_SCHEMA)
        return parse_tf(raw, allow_insufficient=True)
