"""Helper functions for processing entities and URIs."""

from collections import defaultdict

from .dbpedia import lookup_entity_uri


def parse_entity_extraction(extraction: dict) -> tuple[dict, dict]:
    """Parses entity extraction results and maps entity names to URIs and importance values."""
    entity_uris = {}
    entity_importance = {}

    for entity_obj in extraction.get("entities", []):
        if isinstance(entity_obj, dict):
            entity_name = entity_obj.get("name", "")
            entity_type = entity_obj.get("type", "resource")
            importance = entity_obj.get("importance", 1)
        else:
            entity_name = entity_obj
            entity_type = None
            importance = 1

        entity_name = (entity_name or "").strip()
        if not entity_name:
            continue

        uri = lookup_entity_uri(entity_name, entity_type)
        entity_uris[entity_name] = uri
        entity_importance[uri] = max(entity_importance.get(uri, 0), int(importance or 1))

    return entity_uris, entity_importance


def deduplicate_entity_uris(entity_uris: dict) -> dict:
    """Deduplicates entity URIs, grouping names by URI and selecting the longest name as display name."""
    uri_to_names = defaultdict(list)
    for name, uri in entity_uris.items():
        uri_to_names[uri].append(name)

    uri_map = {}
    for uri, names in uri_to_names.items():
        display_name = max(names, key=len)
        uri_map[uri] = display_name

    return uri_map


