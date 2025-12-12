import json
import spacy
from services.gemini_client import call_gemini_api

nlp = spacy.load("en_core_web_sm")

def extract_entities(query: str) -> dict:
    doc = nlp(query)

    # 1️⃣ Named entities (NER) NER = Named Entity Recognition
    entities = [ent.text for ent in doc.ents]

    # 2️⃣ Proper nouns (names, companies, products) PROPN = Proper Nouns
    proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]

    # 3️⃣ Subject nouns nsubj = subject, nsubjpass = subject passive
    subjects = [token.text for token in doc if token.dep_ in ("nsubj", "nsubjpass")]

    # 4️⃣ Fallback: ANY nouns if we still have nothing
    common_nouns = [token.text for token in doc if token.pos_ == "NOUN"]

    # Combine all candidates
    all_entities = list(dict.fromkeys(
        entities + proper_nouns + subjects + common_nouns
    )) or ["UnknownEntity"]

    # Extract verbs
    verbs = [t.lemma_ for t in doc if t.pos_ == "VERB"] or ["unknownVerb"]

    # 🔍 Decide if LLM is needed
    if should_use_llm(all_entities, verbs, query):
        print("➡️ Using LLM to refine entities and expand synonyms...")
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
    response = call_gemini_api(query)
    text = extract_response_text(response)

    try:
        return json.loads(text)
    except:
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