"""Toy Personalized PageRank over the pooled CoLoTa triples.

Seeds default to CoLoTa kg_entities. Set entity_seeds=gss_ner to use GSS llama
NER resource names instead. Unmatched seeds → empty ranking → INSUFFICIENT.
Ranking math is unchanged. Not HippoRAG.
"""

from __future__ import annotations

import networkx as nx

from approaches.rag_transformers.retriever import ranking_metrics
from utils.gss_ner import seed_names
from utils.llm_client import TF_OR_INSUFFICIENT_SCHEMA, chat, parse_tf
from utils.toy_triples import name_matches, parse_triple

PROMPT = """You are given a yes/no question and triples ranked by Personalized PageRank
on a small knowledge graph. The triples may be incomplete or include distractors.
Use only the triples; if they are not enough to decide, answer INSUFFICIENT.

Question: {question}

PPR triples:
{triples}

Return JSON: {{"answer": "TRUE"}}, {{"answer": "FALSE"}}, or {{"answer": "INSUFFICIENT"}}.
"""


class PprApproach:
    name = "ppr"
    needs_corpus = True

    def __init__(self) -> None:
        self.corpus: list[dict] = []
        self.graph = nx.Graph()
        self.alpha = 0.85
        self.config: dict = {}

    def prepare(self, corpus: list[dict], config: dict) -> None:
        cfg = config.get("ppr") or {}
        self.alpha = float(cfg.get("alpha", 0.85))
        self.config = dict(config)
        self.corpus = list(corpus)
        self.graph = nx.Graph()
        for doc in self.corpus:
            parsed = parse_triple(doc.get("triple") or "")
            if parsed is None:
                continue
            subject, _predicate, obj = parsed
            self.graph.add_node(subject)
            self.graph.add_node(obj)
            if subject != obj:
                self.graph.add_edge(subject, obj)

    def rank(self, question: str, item: dict | None = None, **_kwargs) -> list[dict]:
        names = seed_names(item, question, self.config)
        seeds = [
            node
            for node in self.graph.nodes
            if name_matches(node, names)
        ]
        if not seeds or self.graph.number_of_nodes() == 0:
            return []
        personalization = {node: 1.0 / len(seeds) for node in seeds}
        scores = nx.pagerank(
            self.graph,
            alpha=self.alpha,
            personalization=personalization,
            dangling=personalization,
            max_iter=100,
        )
        ranked = []
        for doc in self.corpus:
            parsed = parse_triple(doc.get("triple") or "")
            if parsed is None:
                score = 0.0
            else:
                subject, _predicate, obj = parsed
                score = 0.5 * (scores.get(subject, 0.0) + scores.get(obj, 0.0))
            ranked.append({
                "triple": doc["triple"],
                "text": doc.get("text") or doc["triple"],
                "source_ids": doc.get("source_ids") or [],
                "score": float(score),
            })
        ranked.sort(key=lambda hit: (-hit["score"], hit["triple"]))
        for index, hit in enumerate(ranked, start=1):
            hit["rank"] = index
        return ranked

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
        item: dict | None = None,
        **_kwargs,
    ) -> str:
        hits = (ranking if ranking is not None else self.rank(question, item=item))[:k]
        prompt = PROMPT.format(
            question=question,
            triples="\n".join(
                f"- {hit['triple']}  (score={float(hit.get('score', 0.0)):.3f})"
                for hit in hits
            )
            or "(none)",
        )
        raw = chat(prompt, model=model, schema=TF_OR_INSUFFICIENT_SCHEMA)
        return parse_tf(raw, allow_insufficient=True)
