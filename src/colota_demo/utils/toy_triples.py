"""Parse CoLoTa (s, p, o) strings and match kg_entities onto the pooled corpus."""

from __future__ import annotations

import re

TRIPLE_RE = re.compile(r"^\((.*)\)$")


def parse_triple(text: str) -> tuple[str, str, str] | None:
    raw = (text or "").strip()
    match = TRIPLE_RE.match(raw)
    if not match:
        return None
    parts = match.group(1).split(", ", 2)
    if len(parts) != 3:
        return None
    subject, predicate, obj = (part.strip() for part in parts)
    if not subject or not predicate or not obj:
        return None
    return subject, predicate, obj


def normalize_name(name: str) -> str:
    return " ".join((name or "").strip().lower().replace("_", " ").split())


def entity_names(item: dict | None) -> list[str]:
    entities = (item or {}).get("kg_entities") or {}
    names = []
    seen: set[str] = set()
    for name in entities:
        label = str(name or "").strip()
        key = normalize_name(label)
        if label and key not in seen:
            seen.add(key)
            names.append(label)
    return names


def name_matches(node: str, names: list[str]) -> bool:
    node_key = normalize_name(node)
    if not node_key:
        return False
    return any(node_key == normalize_name(name) for name in names)


def triple_touches_entities(triple: str, names: list[str]) -> bool:
    parsed = parse_triple(triple)
    if parsed is None or not names:
        return False
    subject, _predicate, obj = parsed
    return name_matches(subject, names) or name_matches(obj, names)
