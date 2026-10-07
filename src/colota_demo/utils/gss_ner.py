"""Shared GSS llama NER for toy seed lists.

Calls the original GSS extract_entities() (few-shot Wikidata prompt). Does not
run 1-hop / prune / rank. Resource-type names only become seeds. Cache is
shared across PPR and extract_embed experiment folders.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from utils.dataset import ROOT
from utils.toy_triples import entity_names, normalize_name

DEFAULT_CACHE = "out/colota_demo/gss_ner_cache"


def resolve_seed_source(config: dict | None) -> str:
    raw = str((config or {}).get("entity_seeds") or "gold").strip().lower()
    if raw in {"gold", "kg_entities"}:
        return "gold"
    if raw in {"gss_ner", "ner"}:
        return "gss_ner"
    raise ValueError(f"Unknown entity_seeds={raw!r}; use gold or gss_ner")


def resolve_cache_dir(config: dict | None) -> Path:
    raw = (config or {}).get("ner_cache") or DEFAULT_CACHE
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def resource_names_from_extraction(extraction: dict | None) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for entity in (extraction or {}).get("entities") or []:
        if isinstance(entity, dict):
            kind = str(entity.get("type") or "resource").strip().lower()
            label = str(entity.get("name") or "").strip()
        else:
            kind = "resource"
            label = str(entity or "").strip()
        if kind != "resource" or not label:
            continue
        key = normalize_name(label)
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(label)
    return names


def load_or_extract(
    question: str,
    item_id: str,
    cache_dir: Path,
    model: str | None = None,
) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{item_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    os.environ.setdefault("DATASET", "wikidata")
    os.environ.setdefault("LLM_PROVIDER", "ollama")
    if model:
        os.environ["OLLAMA_MODEL"] = model
    from approaches.gss.utils.parser import extract_entities

    raw = extract_entities(question)
    payload = {
        "id": item_id,
        "question": question,
        "raw": raw,
        "resource_names": resource_names_from_extraction(raw),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def seed_names(item: dict | None, question: str, config: dict | None) -> list[str]:
    source = resolve_seed_source(config)
    if source == "gold":
        return entity_names(item)
    item_id = str((item or {}).get("id") or "unknown")
    payload = load_or_extract(
        question,
        item_id,
        resolve_cache_dir(config),
        model=str((config or {}).get("model") or "") or None,
    )
    return list(payload.get("resource_names") or [])
