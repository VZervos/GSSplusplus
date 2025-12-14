import json
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOPWORDS
from services.gemini_client import call_gemini_api

nlp = spacy.load("en_core_web_sm")

CUSTOM_DROP = {"who", "what", "which", "where", "when", "why", "how"}

def extract_entities(query: str) -> dict:
    doc = nlp(query)

    # 1️⃣ Named entities (NER) NER = Named Entity Recognition
    entities = [ent.text for ent in doc.ents]

    # 2️⃣ Targets of key prepositions (by, in, of) - super informative for DBpedia relations
    pobj_by_in = []
    for tok in doc:
        if tok.dep_ == "prep" and tok.text.lower() in ("by", "in", "of"):
            for child in tok.children:
                if child.dep_ == "pobj":
                    span = doc[child.left_edge.i : child.right_edge.i + 1].text
                    pobj_by_in.append(span)

    # 3️⃣ Noun chunks (multiword phrases)
    noun_chunks = [chunk.text for chunk in doc.noun_chunks]

    # 4️⃣ Proper nouns (names, companies, products) PROPN = Proper Nouns
    proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]

    # 5️⃣ Fallback: ANY nouns if we still have nothing
    common_nouns = [token.text for token in doc if token.pos_ == "NOUN"]

    # Combine candidates in priority order: ents > pobj_by_in > noun_chunks > PROPN > NOUN
    all_entities = list(dict.fromkeys(
        entities + pobj_by_in + noun_chunks + proper_nouns + common_nouns
    )) or ["UnknownEntity"]

    # Extract verbs
    verbs = [t.lemma_ for t in doc if t.pos_ == "VERB"] or ["unknownVerb"]

    # Decide if LLM is needed
    if should_use_llm(all_entities, verbs, query):
        print("-> Using LLM to refine entities and expand synonyms...")
        return refine_with_llm(query)

    # Return spaCy-only extraction
    return {
        "entities": all_entities,
        "verbs": verbs,
        "synonyms": []  # only added by LLM
    }

def should_use_llm(entities, verbs, query):
    """
    Simple condition to check if spaCy extraction is weak.
    """
    # If spaCy found nothing useful
    if entities == ["UnknownEntity"]:
        return True

    # Only 1 generic noun (e.g., "father")
    if len(entities) == 1 and entities[0].lower() in ["father", "mother", "city", "company", "person", "thing"]:
        return True

    # Verb missing
    if verbs == ["unknownVerb"]:
        return True

    # Pronouns usually require LLM interpretation
    if any(p in query.lower() for p in ["his", "her", "their", "its"]):
        return True

    return False

def refine_with_llm(query: str) -> dict:
    prompt = f"""
Extract entities and verbs from the question.

Rules:
- "entities": MUST be exact spans copied from the question text (no paraphrasing).
- "synonyms": MUST be ONLY simple normalization variants of the returned entities.
  Allowed synonym operations:
  - change casing (e.g., "Foo Bar" -> "foo bar")
  - replace spaces with underscores (e.g., "Foo Bar" -> "Foo_Bar")
  - replace underscores with spaces (e.g., "Foo_Bar" -> "Foo Bar")
  - optionally: singular/plural for common nouns (e.g., "boardgame" -> "boardgames")
  Not allowed:
  - inserting/removing punctuation or splitting acronyms (no "G.M.T.", "G M T", "g-m-t")
  - adding extra words or suffixes not present in the entity (no "GMT game")
  - introducing new entities not grounded in the question text
- DO NOT introduce new entities that do not appear in the question.
- Exception (light inference): if the question has a pattern like "X by Y" and Y is an acronym/short label, you may add ONE expanded/canonical form of Y if it is strongly implied (e.g., "GMT" -> "GMT Games").

Return ONLY valid JSON (no markdown, no explanation).

JSON format:
{{
  "entities": ["..."],
  "verbs": ["..."],
  "synonyms": ["..."]
}}

Question: "{query}"
"""

    response = call_gemini_api(prompt)
    text = extract_response_text(response).strip()

    # Clean JSON → sometimes Gemini adds ```json ...``` wrappers
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception as e:
        print("ERROR: LLM JSON parse failed. Text was:", text)
        return {
            "entities": ["UnknownEntity"],
            "verbs": ["unknownVerb"],
            "synonyms": []
        }
        
def extract_response_text(api_response: dict) -> str:
    try:
        return api_response["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        # If structure is different, return formatted JSON
        return json.dumps(api_response, indent=2)

def filter_keywords(keywords):
    filtered_keywords = []
    seen_keywords = set()
    for keyword in keywords:
        cleaned_keyword = (keyword or "").strip()
        if not cleaned_keyword:
            continue

        keyword_lower = cleaned_keyword.lower()

        # dedupe case-insensitive
        if keyword_lower in seen_keywords:
            continue
        seen_keywords.add(keyword_lower)

        # drop interrogatives (always, if single-word)
        if len(cleaned_keyword.split()) == 1 and keyword_lower in CUSTOM_DROP:
            continue

        # drop spaCy stopwords (single-word only)
        if len(cleaned_keyword.split()) == 1 and keyword_lower in SPACY_STOPWORDS:
            continue

        filtered_keywords.append(cleaned_keyword)
    return filtered_keywords