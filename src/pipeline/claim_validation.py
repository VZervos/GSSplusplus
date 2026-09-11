"""Experiment wrapper: build GSS query strings and call stock pipeline() only.

Modes differ only by the natural-language query passed into GSS.
No custom seeding, scoring, or expansion — GSS internals are untouched.
"""

from __future__ import annotations

from pipeline.pipeline import pipeline
from utils.dbpedia import lookup_entity_uri
from config.settings import DATASET, RESOURCE_URI_PREFIX

VALID_MODES = ("answer", "subject", "claim")


def _prefer_kg_uri(explicit_uri: str | None, label: str) -> str | None:
    """Prefer an explicit URI when it matches the active KG; else label lookup."""
    uri = (explicit_uri or "").strip()
    if DATASET == "wikidata":
        if uri.startswith("http://www.wikidata.org/entity/"):
            return uri
        if uri.startswith("wd:"):
            return RESOURCE_URI_PREFIX + uri[3:]
        # LAMA/T-REx bare QIDs, e.g. "Q2942078"
        if len(uri) > 1 and uri[0] in "qQ" and uri[1:].isdigit():
            return f"{RESOURCE_URI_PREFIX}Q{uri[1:]}"
    else:
        if uri.startswith("http://dbpedia.org/resource/"):
            return uri
    name = (label or "").replace(" ", "_").strip()
    if not name:
        return None
    return lookup_entity_uri(name, "resource")


def verbalize_claim(
    question: str,
    subject_label: str,
    answer_text: str,
    relation: dict | None = None,
) -> str:
    """Build a claim string from FactEval/LAMA template when available."""
    sub = (subject_label or "").strip()
    ans = (answer_text or "").strip()
    rel = relation or {}
    template = (rel.get("template") or "").strip()
    if template and sub and ans:
        claim = (
            template.replace("[X]", sub)
            .replace("[Y]", ans)
            .replace(" .", ".")
            .strip()
        )
        if claim:
            return claim
    if sub and ans:
        return f"{sub}: {ans}"
    return ans or question


def build_gss_query(
    mode: str,
    question: str,
    llm_answer: str,
    subject: dict | None = None,
    relation: dict | None = None,
) -> str:
    """Map experiment mode → query string for stock GSS."""
    mode = (mode or "answer").lower().strip()
    subject_label = ((subject or {}).get("label") or "").strip()
    answer = (llm_answer or "").strip()
    claim = verbalize_claim(question, subject_label, answer, relation)

    if mode == "answer":
        # Summary about the claimed answer entity
        return answer or question
    if mode == "subject":
        # Summary about the asked entity
        return subject_label or question
    if mode == "claim":
        return claim
    raise ValueError(f"Unknown mode '{mode}'. Expected one of {VALID_MODES}")


def run_stock_gss(
    *,
    mode: str,
    question: str,
    llm_answer: str,
    subject: dict | None = None,
    relation: dict | None = None,
) -> tuple[list, dict, dict]:
    """Run unmodified GSS on a mode-specific query string.

    Returns:
        (subgraph_triples, uri_importance_map, meta)
    """
    mode = (mode or "answer").lower().strip()
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Expected one of {VALID_MODES}")

    subject = subject or {}
    subject_label = (subject.get("label") or "").strip()
    answer = (llm_answer or "").strip()
    claim_text = verbalize_claim(question, subject_label, answer, relation)
    gss_query = build_gss_query(mode, question, llm_answer, subject, relation)

    # Labels/URIs for post-hoc validation only (not fed into GSS as special seeds)
    answer_name = answer.replace(" ", "_") if answer else ""
    answer_uri = _prefer_kg_uri(None, answer)
    subject_uri = _prefer_kg_uri(subject.get("uri"), subject_label)

    print(f"Starting stock GSS (mode={mode})...")
    print(f"  GSS query: {gss_query}")
    print(f"  Subject (validation): {subject_label!r} -> {subject_uri}")
    print(f"  Answer  (validation): {answer!r} -> {answer_uri}")
    if mode == "claim":
        print(f"  Claim text: {claim_text}")

    if not gss_query.strip():
        return [], {}, {
            "mode": mode,
            "gss_query": gss_query,
            "claim_text": claim_text,
            "answer_entity": {"name": answer_name, "type": "resource", "importance": 5},
            "answer_uri": answer_uri,
            "subject_uri": subject_uri,
            "subject_label": subject_label,
            "relation": relation,
            "error": "empty_gss_query",
        }

    triples, uri_importance_map = pipeline(gss_query)

    meta = {
        "mode": mode,
        "gss_query": gss_query,
        "claim_text": claim_text,
        "answer_entity": {"name": answer_name, "type": "resource", "importance": 5},
        "answer_uri": answer_uri,
        "subject_uri": subject_uri,
        "subject_label": subject_label,
        "relation": relation,
        "entity_uris": {},
    }
    return triples, uri_importance_map, meta
