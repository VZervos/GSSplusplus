import json

from services.gemini_client import call_gemini_api

ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to",
    "was", "were", "will", "with", "the", "this", "but", "they", "have",
    "had", "has", "having", "do", "does", "did", "will", "would", "should",
    "could", "may", "might", "must", "can", "cannot", "i", "you", "we",
    "she", "her", "him", "his", "their", "them", "these", "those", "or",
    "if", "than", "so", "no", "not", "only", "more", "most", "very", "just",
    "all", "each", "both", "few", "many", "some", "such", "own", "same",
    "other", "another", "any", "all", "both", "every", "much", "more",
    "most", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just", "should", "now"
}

CUSTOM_DROP = {"who", "what", "which", "where", "when", "why", "how"}


def extract_entities(query: str) -> dict:
    """
    Extract entities from query using LLM-based extraction.
    All entity extraction is now handled by the Gemini API.
    """
    return refine_with_llm(query)


def refine_with_llm(query: str) -> dict:
    prompt = f"""
Extract entities, verbs, and safe DBpedia lookup variants from the question.

Return ONLY valid JSON (no markdown).

JSON format:
{{
  "entities": [
    {{"name": "...", "type": "resource", "importance": 5}},
    {{"name": "...", "type": "property", "importance": 4}}
  ],
  "verbs": ["..."],        // lemmatized if possible
  "synonyms": ["..."]      // Additional lookup variants (optional)
}}

ENTITY EXTRACTION RULES - BE COMPREHENSIVE:

1. DIRECT ENTITIES (importance: 5):
   - Extract all entities directly mentioned in the question
   - Use DBpedia-friendly format: Title_Case_With_Underscores (e.g., "movies" -> "Film", "Movie")
   - For acronyms/orgs: ALWAYS include both short form AND canonical expansion variants
   - Examples: "IBM" -> include both "IBM" and "IBM_Corporation", "UN" -> include both "UN" and "United_Nations"
   - For company/publisher acronyms in publishing/creation contexts: include both acronym AND common DBpedia variants with suffixes like "_Games", "_Press", "_Publishing", etc.

2. KEY RELATIONSHIPS/PROPERTIES (importance: 4-5):
   - Extract predicates/properties directly implied by the question
   - Patterns: "X by Y" -> include "director", "author", "publisher", or "creator" property (importance: 5)
   - Patterns: "Who created X" -> include "creator" or "author" property (importance: 5)
   - Patterns: "Where was X born" -> include "birthPlace" property (importance: 5)
   - Use DBpedia property names: "director", "author", "creator", "birthPlace", "capital", "publisher", etc.

3. SPECIFIC RELATED CONCEPTS (importance: 3):
   - When a specific concept is mentioned, include related but SPECIFIC DBpedia categories/types
   - Example: "board games" -> include "Board_Game", "Boardgame", "Wargame", "Tabletop_game" (importance: 3)
   - Example: "movies" -> include "Film", "Movie", "Motion_Picture" (importance: 3)
   - AVOID overly generic terms here - prefer specific categories

4. MODERATELY RELATED CONCEPTS (importance: 2):
   - Broader but still relevant categories
   - Example: "Work", "CreativeWork" for creative content questions
   - Example: "Entertainment" for entertainment-related questions
   - Still avoid the most generic terms

5. GENERIC/BACKGROUND CONCEPTS (importance: 1):
   - Only include VERY generic terms if absolutely necessary for retrieval
   - Examples: "Game", "Thing", "Entity" - these should be RARELY used
   - CRITICAL: Avoid assigning importance 2 or higher to generic terms like "Game", "Person", "Place", "Thing", "Entity", "Object"
   - Generic terms should ONLY be used as a last resort when no more specific terms exist

6. IMPORTANCE SCORING (1-5 scale):
   - 5 = Core entities directly mentioned (main subject, key organization/person, essential properties)
   - 4 = Important relationships/properties that are central to the question
   - 3 = Specific related concepts and categories (e.g., "Board_Game" for board game questions)
   - 2 = Moderately related broader concepts (use sparingly)
   - 1 = Very generic/background concepts (use only when necessary, avoid if possible)
   
   CRITICAL RULE: Generic terms like "Game", "Person", "Place", "Thing", "Entity", "Object", "Work" should almost always be importance 1, never 2 or higher.

5. NAME FORMATTING:
   - Use Title_Case_With_Underscores for DBpedia compatibility
   - Keep original text spans when possible, but convert to DBpedia format
   - Examples: "movies" -> "Film" or "Movie", "IBM" -> "IBM" and "IBM_Corporation"
   - For company/publisher acronyms: include both acronym AND full name variant (e.g., "MIT Press" -> "MIT" and "MIT_Press")

SYNONYM RULES (optional, for additional lookup help):
- Include alternative spellings, casing variants, or lookup hints
- These are supplementary to the main entities array

NOT allowed:
- Interrogative pronouns as entities ("who", "what", "where", "when", "why", "how")
- Inventing unrelated famous people/places
- Adding completely unrelated concepts

EXAMPLES:
Question: "What is the capital of France?"
Expected entities:
- {{"name": "France", "type": "resource", "importance": 5}}
- {{"name": "capital", "type": "property", "importance": 5}}
- {{"name": "Country", "type": "resource", "importance": 1}}

Question: "List all movies directed by Christopher Nolan"
Expected entities:
- {{"name": "Christopher_Nolan", "type": "resource", "importance": 5}}
- {{"name": "Movie", "type": "resource", "importance": 3}}
- {{"name": "Film", "type": "resource", "importance": 3}}
- {{"name": "Motion_Picture", "type": "resource", "importance": 3}}
- {{"name": "director", "type": "property", "importance": 5}}
- {{"name": "Work", "type": "resource", "importance": 1}}

Question: "When was the Eiffel Tower built?"
Expected entities:
- {{"name": "Eiffel_Tower", "type": "resource", "importance": 5}}
- {{"name": "completionDate", "type": "property", "importance": 5}}
- {{"name": "openingDate", "type": "property", "importance": 3}}
- {{"name": "Building", "type": "resource", "importance": 1}}

Question: "What is the population of Tokyo?"
Expected entities:
- {{"name": "Tokyo", "type": "resource", "importance": 5}}
- {{"name": "populationTotal", "type": "property", "importance": 5}}
- {{"name": "City", "type": "resource", "importance": 1}}
- {{"name": "Settlement", "type": "resource", "importance": 1}}

Question: "List all books published by MIT Press"
Expected entities:
- {{"name": "MIT_Press", "type": "resource", "importance": 5}}
- {{"name": "MIT", "type": "resource", "importance": 5}}
- {{"name": "Book", "type": "resource", "importance": 3}}
- {{"name": "WrittenWork", "type": "resource", "importance": 2}}
- {{"name": "publisher", "type": "property", "importance": 5}}
- {{"name": "Work", "type": "resource", "importance": 1}}

Question: "List all board games by GMT"
Expected entities:
- {{"name": "GMT_Games", "type": "resource", "importance": 5}}
- {{"name": "GMT", "type": "resource", "importance": 5}}
- {{"name": "publisher", "type": "property", "importance": 5}}
- {{"name": "Board_Game", "type": "resource", "importance": 3}}
- {{"name": "Boardgame", "type": "resource", "importance": 3}}
- {{"name": "Wargame", "type": "resource", "importance": 3}}
- {{"name": "Tabletop_game", "type": "resource", "importance": 3}}
- {{"name": "Game", "type": "resource", "importance": 1}}

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
                    # Ensure importance is in valid range 1-5, default to 1
                    importance = entity.get("importance", 1)
                    if not isinstance(importance, int) or importance < 1 or importance > 5:
                        importance = 1
                    formatted_entities.append({
                        "name": entity.get("name", ""),
                        "type": entity.get("type", "resource"),
                        "importance": importance
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


def filter_keywords(keywords) -> list:
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

        # drop English stopwords (single-word only)
        if len(cleaned_keyword.split()) == 1 and keyword_lower in ENGLISH_STOPWORDS:
            continue

        filtered_keywords.append(cleaned_keyword)
    return filtered_keywords
