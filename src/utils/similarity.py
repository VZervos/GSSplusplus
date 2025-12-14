from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.dbpedia import extract_entity_name_from_uri

# Model for semantic similarity on short texts
model = SentenceTransformer("all-MiniLM-L6-v2")


def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = name.replace("_", " ")
    return name.strip()


def normalize_predicate(name: str) -> str:
    """Normalize predicate names for better semantic matching."""
    name = normalize_name(name)
    return name.lower()


def triple_to_text(triple: dict) -> str:
    """
    Convert a DBpedia triple into a readable natural-language string
    suitable for embedding.
    """
    s_uri = triple.get("s", "")
    p_uri = triple.get("p", "")
    o_uri = triple.get("o", "")

    s_name = normalize_name(extract_entity_name_from_uri(s_uri) or s_uri)
    p_name = normalize_predicate(extract_entity_name_from_uri(p_uri) or p_uri)
    o_name = normalize_name(extract_entity_name_from_uri(o_uri) or o_uri)

    return f"{s_name} {p_name} {o_name}".strip()


def compute_similarity_scores(triples: list, query: str) -> None:
    """
    Compute embedding-based cosine similarity between the query
    and each triple. Adds `triple['similarity']` in range [0, 1].
    """
    if not triples:
        return

    triple_texts = [triple_to_text(t) for t in triples]

    embeddings = model.encode(
        [query] + triple_texts,
        normalize_embeddings=True
    )

    query_emb = embeddings[0]
    triple_embs = embeddings[1:]

    similarities = cosine_similarity(
        query_emb.reshape(1, -1),
        triple_embs
    )[0]

    for triple, sim in zip(triples, similarities):
        # Normalize cosine similarity from [-1, 1] → [0, 1]
        sim = (sim + 1.0) / 2.0
        triple["similarity"] = float(sim)
