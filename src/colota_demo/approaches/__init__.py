"""Registered answering approaches for the CoLoTa demo suite."""

from __future__ import annotations

import importlib

_LOADERS = {
    "llm_only": ("approaches.llm_only", "LlmOnlyApproach"),
    "gold_triples": ("approaches.gold_triples", "GoldTriplesApproach"),
    "rag_transformers": ("approaches.rag_transformers", "RagTransformersApproach"),
    "gss": ("approaches.gss", "GssApproach"),
    "isummary": ("approaches.isummary", "IsummaryApproach"),
    "ppr": ("approaches.ppr", "PprApproach"),
    "extract_embed": ("approaches.extract_embed", "ExtractEmbedApproach"),
}


def known_approaches() -> list[str]:
    return list(_LOADERS)


def get_approach(name: str):
    if name not in _LOADERS:
        known = ", ".join(_LOADERS)
        raise KeyError(f"Unknown approach {name!r}. Known: {known}")
    module_name, class_name = _LOADERS[name]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()
