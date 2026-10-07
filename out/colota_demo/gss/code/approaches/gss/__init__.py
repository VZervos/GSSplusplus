"""Original GSS: live Wikidata subgraph, then a separate answering prompt.

Runs the restored pipeline (NER → SPARQL 1-hop → prune/score/rank) and feeds
the ranked triples to the LLM at the same k depths as RAG.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from utils.llm_client import TF_OR_INSUFFICIENT_SCHEMA, chat, parse_tf

from .config import settings

os.environ.setdefault("DATASET", "wikidata")
os.environ.setdefault("LLM_PROVIDER", "ollama")

PROMPT = """You are given a yes/no question and a GSS query-focused subgraph.
The triples may be incomplete or include distractors. Use only the triples;
if they are not enough to decide, answer INSUFFICIENT.

Question: {question}

GSS triples:
{triples}

Return JSON: {{"answer": "TRUE"}}, {{"answer": "FALSE"}}, or {{"answer": "INSUFFICIENT"}}.
"""


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return float(value)
    return str(value)


def format_triple(triple: dict) -> str:
    subject = triple.get("s_label") or triple.get("s") or ""
    predicate = triple.get("p_label") or triple.get("p") or ""
    obj = triple.get("o_label") or triple.get("o") or ""
    return f"({subject}, {predicate}, {obj})"


class GssApproach:
    name = "gss"
    needs_corpus = False

    def __init__(self) -> None:
        self.cache_dir: Path | None = None
        self._pipeline = None
        self.a = 0.75

    def prepare(self, corpus: list[dict], config: dict) -> None:
        model = config.get("model")
        if model:
            os.environ["OLLAMA_MODEL"] = model
        gss_cfg = config.get("gss") or {}
        self.a = float(gss_cfg.get("a", 0.75))
        settings.A = self.a
        settings.SCORING_IMPORTANCE_WEIGHT = self.a
        settings.SCORING_SIMILARITY_WEIGHT = 1.0 - self.a
        out_dir = Path(config.get("out_dir") or "out/colota_demo")
        if not out_dir.is_absolute():
            from utils.dataset import ROOT

            out_dir = ROOT / out_dir
        self.cache_dir = out_dir / "gss_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_pipeline(self):
        if self._pipeline is None:
            from .pipeline.pipeline import pipeline

            self._pipeline = pipeline
        return self._pipeline

    def summarize(self, question: str, item_id: str | None = None) -> list[dict]:
        cached = self._read_cache(item_id)
        if cached is not None:
            return cached
        pipeline = self._load_pipeline()
        triples, _uri_map = pipeline(question)
        ranked = sorted(
            triples,
            key=lambda triple: float(triple.get("final_score") or 0.0),
            reverse=True,
        )
        ranked = _jsonable(ranked)
        self._write_cache(item_id, ranked)
        return ranked

    def predict(
        self,
        question: str,
        *,
        model: str,
        k: int,
        ranking: list[dict] | None = None,
        item_id: str | None = None,
        **_kwargs,
    ) -> str:
        hits = (ranking if ranking is not None else self.summarize(question, item_id))[:k]
        prompt = PROMPT.format(
            question=question,
            triples="\n".join(
                f"- {format_triple(hit)}"
                + (
                    f"  (score={float(hit.get('final_score', 0.0)):.3f})"
                    if hit.get("final_score") is not None
                    else ""
                )
                for hit in hits
            )
            or "(none)",
        )
        raw = chat(prompt, model=model, schema=TF_OR_INSUFFICIENT_SCHEMA)
        return parse_tf(raw, allow_insufficient=True)

    def _cache_path(self, item_id: str | None) -> Path | None:
        if not self.cache_dir or not item_id:
            return None
        return self.cache_dir / f"{item_id}_a{self.a}.json"

    def _read_cache(self, item_id: str | None) -> list[dict] | None:
        path = self._cache_path(item_id)
        if path is None or not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def _write_cache(self, item_id: str | None, triples: list[dict]) -> None:
        path = self._cache_path(item_id)
        if path is None:
            return
        path.write_text(json.dumps(triples, indent=2, ensure_ascii=False), encoding="utf-8")
