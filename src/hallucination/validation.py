"""Post-hoc validation of an LLM answer against a stock-GSS ACE subgraph.

GSS is treated as a black box. All support/reject logic lives here.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import unquote

# FactEval/LAMA relation → predicate local-name allowlist (DBpedia names + Wikidata P-ids).
# Matching is case-insensitive on URI local names (property/, ontology/, wdt:P*).
RELATION_PREDICATE_ALLOWLIST: dict[str, set[str]] = {
    "P19": {"birthplace", "placeofbirth", "bornin", "p19"},
    "P20": {"deathplace", "placeofdeath", "diedin", "p20"},
    "P27": {"nationality", "citizenship", "country", "countryofcitizenship", "p27"},
    "P36": {"capital", "p36"},
    "P106": {"occupation", "profession", "p106"},
    "P17": {"country", "locationcountry", "ispartof", "p17"},
    "P159": {"headquarter", "headquarters", "locationcity", "location", "city", "p159"},
    "P495": {"country", "countryoforigin", "origin", "p495"},
    "P740": {"hometown", "foundinglocation", "location", "formedin", "foundationplace", "p740"},
    "P937": {"worklocation", "workplace", "location", "employer", "p937"},
}

# Predicates that can justify "broader location" (city ⊂ country, etc.).
CONTAINMENT_PREDICATE_KEYS: set[str] = {
    "country",
    "locationcountry",
    "ispartof",
    "location",
    "locatedinarea",
    "partof",
    "department",
    "region",
    "state",
    "province",
    "district",
    "city",
    "capital",  # capital links city↔country in either direction in some graphs
    "p17",
    "p131",
    "p276",
    "p361",
}

# Relations where multi-valued / hierarchical gold is meaningful.
GRANULARITY_RELATIONS: set[str] = {
    "P19", "P20", "P17", "P27", "P159", "P495", "P740", "P937",
}


def normalize_label(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = unquote(text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def labels_match(a: str, b: str) -> bool:
    na, nb = normalize_label(a), normalize_label(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return False
    return len(ta & tb) / max(len(ta), len(tb)) >= 0.8


def uri_local_name(uri: str) -> str:
    if not uri:
        return ""
    return unquote(uri.rstrip("/").split("/")[-1].replace("_", " "))


def predicate_key(uri: str) -> str:
    return re.sub(r"[^a-z0-9]", "", uri_local_name(uri).lower())


def collect_subgraph_labels(triples: list) -> set[str]:
    labels = set()
    for t in triples:
        for key in ("s", "p", "o"):
            labels.add(normalize_label(uri_local_name(t.get(key, ""))))
        for key in ("s_label", "p_label", "o_label", "label"):
            if t.get(key):
                labels.add(normalize_label(str(t[key])))
    return {x for x in labels if x}


def entity_in_subgraph(entity_label: str, entity_uri: str | None, triples: list) -> bool:
    labels = collect_subgraph_labels(triples)
    if entity_uri:
        target = normalize_label(uri_local_name(entity_uri))
        if target and any(labels_match(target, lab) for lab in labels):
            return True
        for t in triples:
            for key in ("s", "p", "o"):
                if t.get(key) == entity_uri:
                    return True
    if entity_label:
        return any(labels_match(entity_label, lab) for lab in labels)
    return False


def answer_matches_gold(llm_answer: str, gold_answers: list[dict]) -> bool:
    for g in gold_answers or []:
        if labels_match(llm_answer, g.get("label", "")):
            return True
        if labels_match(llm_answer, uri_local_name(g.get("uri", ""))):
            return True
    return False


def _mentions(t: dict, label: str, uri: str | None) -> bool:
    nodes = [t.get("s", ""), t.get("p", ""), t.get("o", "")]
    if uri and uri in nodes:
        return True
    labs = [normalize_label(uri_local_name(n)) for n in nodes]
    return bool(label) and any(labels_match(label, lab) for lab in labs if lab)


def entities_linked_in_triple(
    label_a: str,
    uri_a: str | None,
    label_b: str,
    uri_b: str | None,
    triples: list,
) -> bool:
    for t in triples or []:
        if _mentions(t, label_a, uri_a) and _mentions(t, label_b, uri_b):
            return True
    return False


def relation_id(relation: dict | None) -> str | None:
    if not relation:
        return None
    rid = relation.get("relation") or relation.get("predicate_id")
    return str(rid).strip().upper() if rid else None


def predicate_matches_relation(pred_uri: str, relation: dict | None) -> bool:
    """True if predicate is in the allowlist for this FactEval relation."""
    rid = relation_id(relation)
    if not rid:
        return True  # no metadata → do not filter by predicate
    allowed = RELATION_PREDICATE_ALLOWLIST.get(rid)
    if not allowed:
        return True
    return predicate_key(pred_uri) in allowed


def find_supporting_triples(
    subject_label: str,
    subject_uri: str | None,
    answer_label: str,
    answer_uri: str | None,
    triples: list,
    relation: dict | None,
) -> list[dict]:
    """Triples that mention both entities and (when known) a relation-compatible predicate."""
    hits = []
    for t in triples or []:
        if not (
            _mentions(t, subject_label, subject_uri)
            and _mentions(t, answer_label, answer_uri)
        ):
            continue
        if predicate_matches_relation(t.get("p", ""), relation):
            hits.append(t)
    return hits


def find_linked_triples(
    subject_label: str,
    subject_uri: str | None,
    answer_label: str,
    answer_uri: str | None,
    triples: list,
) -> list[dict]:
    """Any triples that mention both entities (predicate unrestricted)."""
    return [
        t
        for t in (triples or [])
        if _mentions(t, subject_label, subject_uri)
        and _mentions(t, answer_label, answer_uri)
    ]


def _gold_labels_and_uris(gold_answers: list[dict]) -> list[tuple[str, str | None]]:
    out = []
    for g in gold_answers or []:
        label = g.get("label") or ""
        uri = g.get("uri")
        uri_s = str(uri).strip() if uri else ""
        # Full KG URIs
        if uri_s.startswith("http://dbpedia.org/") or uri_s.startswith(
            "http://www.wikidata.org/"
        ):
            out.append((label, uri_s))
        # LAMA bare Wikidata QIDs (e.g. Q456)
        elif len(uri_s) > 1 and uri_s[0] in "qQ" and uri_s[1:].isdigit():
            out.append((label, f"http://www.wikidata.org/entity/Q{uri_s[1:]}"))
        else:
            out.append((label, None))
    return out


def assess_granularity(
    *,
    subject_label: str,
    subject_uri: str | None,
    answer_label: str,
    answer_uri: str | None,
    gold_answers: list[dict],
    triples: list,
    relation: dict | None,
    answer_relation_support: bool,
) -> dict:
    """Detect multi-valued / hierarchical agreement with gold (outside exact string match).

    Cases:
      multi_valued_relation:
        subject -[rel]-> llm_answer AND subject -[rel]-> gold (e.g. birthPlace France & Lyon)
      hierarchical_containment:
        gold and llm_answer co-occur on a containment-like predicate in the ACE
        (e.g. Lyon -country-> France), and llm answer also has relation support to subject
    """
    rid = relation_id(relation)
    empty = {
        "granularity_match": False,
        "granularity_type": None,
        "gold_relation_linked": False,
        "gold_answer_containment": False,
    }
    if not triples or not answer_label or not gold_answers:
        return empty
    if rid and rid not in GRANULARITY_RELATIONS:
        return empty

    gold_rel_hits = []
    for g_label, g_uri in _gold_labels_and_uris(gold_answers):
        if not g_label:
            continue
        gold_rel_hits.extend(
            find_supporting_triples(
                subject_label, subject_uri, g_label, g_uri, triples, relation
            )
        )

    gold_relation_linked = len(gold_rel_hits) > 0

    # Hierarchical: gold ↔ claimed answer via containment predicate
    containment_hits = []
    for g_label, g_uri in _gold_labels_and_uris(gold_answers):
        if not g_label:
            continue
        for t in triples:
            if not (
                _mentions(t, g_label, g_uri)
                and _mentions(t, answer_label, answer_uri)
            ):
                continue
            if predicate_key(t.get("p", "")) in CONTAINMENT_PREDICATE_KEYS:
                containment_hits.append(t)

    gold_answer_containment = len(containment_hits) > 0

    granularity_type = None
    if answer_relation_support and gold_relation_linked:
        granularity_type = "multi_valued_relation"
    elif answer_relation_support and gold_answer_containment:
        granularity_type = "hierarchical_containment"

    return {
        "granularity_match": granularity_type is not None,
        "granularity_type": granularity_type,
        "gold_relation_linked": gold_relation_linked,
        "gold_answer_containment": gold_answer_containment,
        "n_gold_relation_triples": len(gold_rel_hits),
        "n_containment_triples": len(containment_hits),
    }


def validate_with_summary(
    question: str,
    llm_answer: str,
    gold_answers: list[dict],
    subject: dict | None,
    triples: list,
    meta: dict,
    mode: str | None = None,
    relation: dict | None = None,
) -> dict:
    """Decide supported / weak_support / unsupported / uncertain from the ACE.

    - supported: subject+answer linked with relation-compatible predicate
    - weak_support: subject+answer linked, but predicate not in relation allowlist
    - unsupported / uncertain: otherwise

    Gold scoring:
    - gold_match: exact/fuzzy label match
    - gold_acceptable: gold_match OR granularity_match (multi-valued / hierarchical)
    Detection uses gold_acceptable so coarser KG-true answers are not false_support.
    """
    mode = (mode or meta.get("mode") or "answer").lower().strip()
    relation = relation or meta.get("relation")
    gold_match = answer_matches_gold(llm_answer, gold_answers)

    answer_uri = meta.get("answer_uri")
    answer_entity = meta.get("answer_entity") or {}
    error = meta.get("error")

    subject_label = (subject or {}).get("label") or meta.get("subject_label") or ""
    subject_uri = meta.get("subject_uri")
    answer_label = answer_entity.get("name") or llm_answer

    subject_in_graph = entity_in_subgraph(subject_label, subject_uri, triples)
    answer_in_graph = entity_in_subgraph(answer_label, answer_uri, triples)

    linked_triples = find_linked_triples(
        subject_label, subject_uri, answer_label, answer_uri, triples
    )
    linked_any = len(linked_triples) > 0
    supporting = find_supporting_triples(
        subject_label, subject_uri, answer_label, answer_uri, triples, relation
    )
    relation_linked = len(supporting) > 0

    gran = assess_granularity(
        subject_label=subject_label,
        subject_uri=subject_uri,
        answer_label=answer_label,
        answer_uri=answer_uri,
        gold_answers=gold_answers,
        triples=triples,
        relation=relation,
        answer_relation_support=relation_linked,
    )
    gold_acceptable = bool(gold_match or gran["granularity_match"])

    rid = relation_id(relation)
    if error or not (llm_answer or "").strip():
        decision = "uncertain"
        reason = error or "empty_llm_answer"
    elif not triples:
        decision = "uncertain"
        reason = "empty_subgraph"
    elif relation_linked:
        decision = "supported"
        reason = "subject_answer_linked_with_relation_predicate"
    elif linked_any and rid in RELATION_PREDICATE_ALLOWLIST:
        decision = "weak_support"
        reason = "subject_answer_linked_but_predicate_mismatch"
    elif linked_any:
        decision = "supported"
        reason = "subject_answer_linked_no_relation_filter"
    else:
        decision = "unsupported"
        reason = "no_subject_answer_relation_evidence_in_ace"

    # Detection vs gold (analysis only)
    if decision == "uncertain":
        detection = "uncertain"
    elif gold_acceptable and decision == "supported":
        detection = (
            "true_support"
            if gold_match
            else "true_support_granularity"
        )
    elif gold_acceptable and decision == "weak_support":
        detection = "weak_true_support"
    elif gold_acceptable and decision == "unsupported":
        detection = "missed_support"
    elif (not gold_acceptable) and decision == "supported":
        detection = "false_support"
    elif (not gold_acceptable) and decision == "weak_support":
        detection = "weak_false_support"
    else:
        detection = "not_validated"

    return {
        "mode": mode,
        "decision": decision,
        "reason": reason,
        "detection": detection,
        "gold_match": gold_match,
        "gold_acceptable": gold_acceptable,
        "granularity_match": gran["granularity_match"],
        "granularity_type": gran["granularity_type"],
        "gold_relation_linked": gran["gold_relation_linked"],
        "gold_answer_containment": gran["gold_answer_containment"],
        "subject_in_graph": subject_in_graph,
        "gold_in_graph": any(
            entity_in_subgraph(g.get("label", ""), None, triples)
            for g in (gold_answers or [])
        ),
        "answer_in_graph": answer_in_graph,
        "subject_answer_linked": linked_any,
        "relation_linked": relation_linked,
        "n_supporting_triples": len(supporting),
        "n_linked_triples": len(linked_triples),
        "supporting_predicates": [t.get("p") for t in supporting[:10]],
        "linked_predicates": [t.get("p") for t in linked_triples[:10]],
        "n_triples": len(triples),
        "answer_uri": answer_uri,
        "answer_entity": answer_entity,
        "subject_uri": subject_uri,
        "relation_id": rid,
        "gss_query": meta.get("gss_query"),
        "claim_text": meta.get("claim_text"),
        "question": question,
        "llm_answer": llm_answer,
    }
