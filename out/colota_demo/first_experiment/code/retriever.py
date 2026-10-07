"""Tiny sentence-transformers retriever over the pooled CoLoTa triple corpus."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer, util

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_KS = (1, 3, 5, 10)


class TripleRetriever:
    def __init__(
        self,
        corpus: list[dict],
        model_name: str = DEFAULT_EMBED_MODEL,
    ):
        self.corpus = corpus
        self.model = SentenceTransformer(model_name)
        texts = [doc["text"] for doc in corpus]
        self.embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        top_k = len(self.corpus) if k is None else min(k, len(self.corpus))
        query_emb = self.model.encode(
            question,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        hits = util.semantic_search(query_emb, self.embeddings, top_k=top_k)[0]
        results = []
        for hit in hits:
            doc = self.corpus[hit["corpus_id"]]
            results.append({
                "triple": doc["triple"],
                "text": doc["text"],
                "source_ids": doc["source_ids"],
                "score": float(hit["score"]),
                "rank": len(results) + 1,
            })
        return results


def recall_at_k(gold_triples: list[str], retrieved: list[dict]) -> float:
    if not gold_triples:
        return 0.0
    retrieved_set = {item["triple"] for item in retrieved}
    hits = sum(1 for triple in gold_triples if triple in retrieved_set)
    return hits / len(gold_triples)


def ranking_metrics(
    gold_triples: list[str],
    ranking: list[dict],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict:
    """Ranks are 1-indexed over the full corpus ranking."""
    rank_by_triple = {item["triple"]: item["rank"] for item in ranking}
    score_by_triple = {item["triple"]: item["score"] for item in ranking}

    gold_ranks = []
    ranks: list[int] = []
    reciprocal = []
    for triple in gold_triples:
        rank = rank_by_triple.get(triple)
        gold_ranks.append({
            "triple": triple,
            "rank": rank,
            "score": score_by_triple.get(triple),
        })
        if rank is None:
            reciprocal.append(0.0)
        else:
            ranks.append(rank)
            reciprocal.append(1.0 / rank)

    first = min(ranks) if ranks else None
    recalls = {
        k: recall_at_k(gold_triples, ranking[:k])
        for k in ks
    }
    return {
        "mrr": (1.0 / first) if first else 0.0,
        "mean_rr": (sum(reciprocal) / len(reciprocal)) if reciprocal else 0.0,
        "first_relevant_rank": first,
        "gold_ranks": gold_ranks,
        "recall": recalls,
    }
