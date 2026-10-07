"""Toy entity-filtered MiniLM retrieval over the pooled CoLoTa triples.

Keeps corpus triples whose subject or object matches seed names, then ranks
that subset with MiniLM cosine (same encoder as RAG). Seeds default to CoLoTa
kg_entities; set entity_seeds=gss_ner for GSS llama NER resource names.
Empty filter → empty ranking → INSUFFICIENT. Ranking math is unchanged.
Not G-Retriever / HippoRAG.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer, util

from approaches.rag_transformers.retriever import DEFAULT_EMBED_MODEL, ranking_metrics
from utils.gss_ner import seed_names
from utils.llm_client import TF_OR_INSUFFICIENT_SCHEMA, chat, parse_tf
from utils.toy_triples import triple_touches_entities

PROMPT = """You are given a yes/no question and triples retrieved after filtering
to entities mentioned in the question, then ranking by embedding similarity.
The triples may be incomplete or include distractors. Use only the triples;
if they are not enough to decide, answer INSUFFICIENT.

Question: {question}

Retrieved triples:
{triples}

Return JSON: {{"answer": "TRUE"}}, {{"answer": "FALSE"}}, or {{"answer": "INSUFFICIENT"}}.
"""


class ExtractEmbedApproach:
    name = "extract_embed"
    needs_corpus = True

    def __init__(self) -> None:
        self.corpus: list[dict] = []
        self.model: SentenceTransformer | None = None
        self.embeddings = None
        self.config: dict = {}

    def prepare(self, corpus: list[dict], config: dict) -> None:
        embed_model = config.get("embed_model") or DEFAULT_EMBED_MODEL
        self.config = dict(config)
        self.corpus = list(corpus)
        self.model = SentenceTransformer(embed_model)
        texts = [doc.get("text") or doc["triple"] for doc in self.corpus]
        self.embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

    def rank(self, question: str, item: dict | None = None, **_kwargs) -> list[dict]:
        if self.model is None or self.embeddings is None:
            raise RuntimeError("extract_embed.prepare() was not called")
        names = seed_names(item, question, self.config)
        indices = [
            index
            for index, doc in enumerate(self.corpus)
            if triple_touches_entities(doc.get("triple") or "", names)
        ]
        if not indices:
            return []
        query_emb = self.model.encode(
            question,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        subset = self.embeddings[indices]
        hits = util.semantic_search(query_emb, subset, top_k=len(indices))[0]
        results = []
        for hit in hits:
            doc = self.corpus[indices[hit["corpus_id"]]]
            results.append({
                "triple": doc["triple"],
                "text": doc.get("text") or doc["triple"],
                "source_ids": doc.get("source_ids") or [],
                "score": float(hit["score"]),
                "rank": len(results) + 1,
            })
        return results

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
