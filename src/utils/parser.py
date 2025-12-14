import json
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOPWORDS
from services.gemini_client import call_gemini_api

nlp = spacy.load("en_core_web_sm")

CUSTOM_DROP = {"who", "what", "which", "where", "when", "why", "how"}

def extract_entities(query: str) -> dict:
    doc = nlp(query)
    return refine_with_llm(query)
    # 1️⃣ Named entities (NER) - highest priority
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
    
    # Filter out interrogative pronouns
    all_entities = [e for e in all_entities if e.lower() not in CUSTOM_DROP]

    # Extract verbs
    verbs = [t.lemma_ for t in doc if t.pos_ == "VERB"] or ["unknownVerb"]

    # Decide if LLM is needed
    if should_use_llm(all_entities, verbs, query):
        print("-> Using LLM to refine entities and expand synonyms...")
        return refine_with_llm(query)

    # Return spaCy-only extraction (convert to structured format)
    formatted_entities = [
        {"name": entity, "type": "resource", "importance": 1}
        for entity in all_entities
    ]
    return {
        "entities": formatted_entities,
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
Extract entities, verbs, and safe DBpedia lookup variants from the question.

Return ONLY valid JSON (no markdown).

JSON format:
{{
  "entities": [
    {{"name": "...", "type": "resource", "importance": 3}},
    {{"name": "...", "type": "property", "importance": 2}}
  ],
  "verbs": ["..."],        // lemmatized if possible
  "synonyms": ["..."]      // Additional lookup variants (optional)
}}

ENTITY EXTRACTION RULES - BE COMPREHENSIVE:

1. DIRECT ENTITIES (importance: 3):
   - Extract all entities directly mentioned in the question
   - Use DBpedia-friendly format: Title_Case_With_Underscores (e.g., "boardgames" -> "Boardgame", "Board_Game")
   - For acronyms/orgs: include both short form AND canonical expansion (e.g., "GMT" -> include both "GMT" and "GMT_Games")

2. RELATED CONCEPTS (importance: 2):
   - When a concept is mentioned, include related DBpedia categories/types
   - Example: "boardgames" -> include "Board_Game", "Boardgame", "Wargame", "Tabletop_game", "Game"
   - Example: "developed" -> include related concepts if contextually relevant

3. IMPLIED PROPERTIES/RELATIONS (importance: 2-3):
   - Extract predicates/properties implied by verbs or question patterns
   - Patterns: "X by Y" -> include "publisher" or "developer" property
   - Patterns: "Who developed X" -> include "developer" property
   - Patterns: "Where was X born" -> include "birthPlace" property
   - Use DBpedia property names: "publisher", "developer", "author", "birthPlace", etc.

4. IMPORTANCE SCORING:
   - 3 = Core entities directly mentioned (main subject, key organization/person)
   - 2 = Related concepts, categories, or implied relations
   - 1 = Background/generic concepts

5. NAME FORMATTING:
   - Use Title_Case_With_Underscores for DBpedia compatibility
   - Keep original text spans when possible, but convert to DBpedia format
   - Examples: "boardgames" -> "Boardgame" or "Board_Game", "GMT" -> "GMT" and "GMT_Games"

SYNONYM RULES (optional, for additional lookup help):
- Include alternative spellings, casing variants, or lookup hints
- These are supplementary to the main entities array

NOT allowed:
- Interrogative pronouns as entities ("who", "what", "where", "when", "why", "how")
- Inventing unrelated famous people/places
- Adding completely unrelated concepts

EXAMPLES:
Question: "List all boardgames by GMT"
Expected entities:
- {{"name": "GMT", "type": "resource", "importance": 3}}
- {{"name": "GMT_Games", "type": "resource", "importance": 3}}
- {{"name": "Boardgame", "type": "resource", "importance": 2}}
- {{"name": "Board_Game", "type": "resource", "importance": 2}}
- {{"name": "Wargame", "type": "resource", "importance": 2}}
- {{"name": "Tabletop_game", "type": "resource", "importance": 2}}
- {{"name": "publisher", "type": "property", "importance": 3}}
- {{"name": "Game", "type": "resource", "importance": 1}}

Question: "Who developed Skype?"
Expected entities:
- {{"name": "Skype", "type": "resource", "importance": 3}}
- {{"name": "developer", "type": "property", "importance": 3}}
- {{"name": "author", "type": "property", "importance": 2}}

Question: "{query}"
"""

    response = call_gemini_api(prompt)
    text = extract_response_text(response).strip()

    # Clean JSON → sometimes Gemini adds ```json ...``` wrappers
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(text)
        # Ensure entities are in the correct format
        if "entities" in result and result["entities"]:
            formatted_entities = []
            for entity in result["entities"]:
                if isinstance(entity, dict):
                    # Already in correct format, ensure required fields
                    formatted_entities.append({
                        "name": entity.get("name", ""),
                        "type": entity.get("type", "resource"),
                        "importance": entity.get("importance", 1)
                    })
                else:
                    # Legacy string format, convert to dict
                    formatted_entities.append({
                        "name": entity,
                        "type": "resource",
                        "importance": 1
                    })
            result["entities"] = formatted_entities
        return result
    except Exception as e:
        print("ERROR: LLM JSON parse failed. Text was:", text)
        return {
            "entities": [{"name": "UnknownEntity", "type": "resource", "importance": 1}],
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