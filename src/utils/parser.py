import json

from config.settings import DATASET
from services.llm_client import call_llm_api, extract_response_text as llm_extract_response_text

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
    return refine_with_llm(query)


def refine_with_llm(query: str) -> dict:
    dataset_name = "Wikidata" if DATASET == "wikidata" else "DBpedia"
    dataset_format = "Q/P numbers (e.g., Q30 for United States, P31 for 'instance of')" if DATASET == "wikidata" else "Title_Case_With_Underscores (e.g., 'movies' -> 'Film', 'Movie')"
    dataset_example = "Q30" if DATASET == "wikidata" else "France"
    dataset_property_example = "P31" if DATASET == "wikidata" else "director"
    
    prompt = f"""
Extract entities, verbs, and safe {dataset_name} lookup variants from the question.

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
   - Use {dataset_name}-friendly format: {dataset_format}
   - For {dataset_name}: {"Use Q numbers for entities (e.g., Q30 for United States) and P numbers for properties (e.g., P31 for 'instance of'). If you don't know the exact Q/P number, use the entity name in a format that can be looked up." if DATASET == "wikidata" else "Use Title_Case_With_Underscores (e.g., 'movies' -> 'Film', 'Movie')"}
   - For acronyms/orgs: ALWAYS include both short form AND canonical expansion variants
   - Examples: {"'IBM' -> include both 'Q312' (if known) or 'IBM' and 'IBM_Corporation', 'UN' -> include both 'Q1065' (if known) or 'UN' and 'United_Nations'" if DATASET == "wikidata" else "'IBM' -> include both 'IBM' and 'IBM_Corporation', 'UN' -> include both 'UN' and 'United_Nations'"}
   - For company/publisher acronyms in publishing/creation contexts: include both acronym AND common variants with suffixes like "_Games", "_Press", "_Publishing", etc.

2. KEY RELATIONSHIPS/PROPERTIES (importance: 4-5):
   - Extract predicates/properties directly implied by the question
   - Patterns: "X by Y" -> include "director", "author", "publisher", or "creator" property (importance: 5)
   - Patterns: "Who created X" -> include "creator" or "author" property (importance: 5)
   - Patterns: "Where was X born" -> include "birthPlace" property (importance: 5)
   - Use {dataset_name} property names: {"P31 (instance of), P27 (country of citizenship), P569 (date of birth), P19 (place of birth), P50 (author), P57 (director), P123 (publisher), etc." if DATASET == "wikidata" else "'director', 'author', 'creator', 'birthPlace', 'capital', 'publisher', etc."}

3. SPECIFIC RELATED CONCEPTS (importance: 3):
   - When a specific concept is mentioned, include related but SPECIFIC {dataset_name} categories/types
   - Example: "board games" -> {"include Q11019 (board game), Q131894 (tabletop game) if known, or 'Board_Game', 'Boardgame', 'Wargame', 'Tabletop_game'" if DATASET == "wikidata" else "include 'Board_Game', 'Boardgame', 'Wargame', 'Tabletop_game'"} (importance: 3)
   - Example: "movies" -> {"include Q11424 (film) if known, or 'Film', 'Movie', 'Motion_Picture'" if DATASET == "wikidata" else "include 'Film', 'Movie', 'Motion_Picture'"} (importance: 3)
   - AVOID overly generic terms here - prefer specific categories

4. MODERATELY RELATED CONCEPTS (importance: 2):
   - Broader but still relevant categories
   - Example: {"Q17537576 (creative work) or 'Work', 'CreativeWork'" if DATASET == "wikidata" else "'Work', 'CreativeWork'"} for creative content questions
   - Example: {"Q7889 (entertainment) or 'Entertainment'" if DATASET == "wikidata" else "'Entertainment'"} for entertainment-related questions
   - Still avoid the most generic terms

5. GENERIC/BACKGROUND CONCEPTS (importance: 1):
   - Only include VERY generic terms if absolutely necessary for retrieval
   - Examples: {"Q11424 (film), Q11410 (game), Q35120 (entity) if known, or 'Game', 'Thing', 'Entity'" if DATASET == "wikidata" else "'Game', 'Thing', 'Entity'"} - these should be RARELY used
   - CRITICAL: Avoid assigning importance 2 or higher to generic terms like "Game", "Person", "Place", "Thing", "Entity", "Object"
   - Generic terms should ONLY be used as a last resort when no more specific terms exist

6. IMPORTANCE SCORING (1-5 scale):
   - 5 = Core entities directly mentioned (main subject, key organization/person, essential properties)
   - 4 = Important relationships/properties that are central to the question
   - 3 = Specific related concepts and categories (e.g., {"'Q11019' (board game) or 'Board_Game'" if DATASET == "wikidata" else "'Board_Game'"} for board game questions)
   - 2 = Moderately related broader concepts (use sparingly)
   - 1 = Very generic/background concepts (use only when necessary, avoid if possible)
   
   CRITICAL RULE: Generic terms like "Game", "Person", "Place", "Thing", "Entity", "Object", "Work" should almost always be importance 1, never 2 or higher.

7. NAME FORMATTING:
   - Use {dataset_format} for {dataset_name} compatibility
   - Keep original text spans when possible, but convert to {dataset_name} format
   - Examples: {"'movies' -> 'Q11424' (if known) or 'Film'/'Movie', 'IBM' -> 'Q312' (if known) or 'IBM' and 'IBM_Corporation'" if DATASET == "wikidata" else "'movies' -> 'Film' or 'Movie', 'IBM' -> 'IBM' and 'IBM_Corporation'"}
   - For company/publisher acronyms: include both acronym AND full name variant (e.g., "MIT Press" -> {"'Q29133' (if known) or 'MIT' and 'MIT_Press'" if DATASET == "wikidata" else "'MIT' and 'MIT_Press'"})

SYNONYM RULES (optional, for additional lookup help):
- Include alternative spellings, casing variants, or lookup hints
- These are supplementary to the main entities array

NOT allowed:
- Interrogative pronouns as entities ("who", "what", "where", "when", "why", "how")
- Inventing unrelated famous people/places
- Adding completely unrelated concepts

EXAMPLES:
Question: "What is the capital of {dataset_example}?"
Expected entities:
- {{"name": "{dataset_example}", "type": "resource", "importance": 5}}
- {{"name": {"P36" if DATASET == "wikidata" else "'capital'"}, "type": "property", "importance": 5}}
- {{"name": {"Q6256" if DATASET == "wikidata" else "'Country'"}, "type": "resource", "importance": 1}}

Question: "List all movies directed by Christopher Nolan"
Expected entities:
- {{"name": {"Q2513" if DATASET == "wikidata" else "'Christopher_Nolan'"}, "type": "resource", "importance": 5}}
- {{"name": {"Q11424" if DATASET == "wikidata" else "'Movie'"}, "type": "resource", "importance": 3}}
- {{"name": {"Q11424" if DATASET == "wikidata" else "'Film'"}, "type": "resource", "importance": 3}}
- {{"name": {"Q11424" if DATASET == "wikidata" else "'Motion_Picture'"}, "type": "resource", "importance": 3}}
- {{"name": {"P57" if DATASET == "wikidata" else "'director'"}, "type": "property", "importance": 5}}
- {{"name": {"Q17537576" if DATASET == "wikidata" else "'Work'"}, "type": "resource", "importance": 1}}

Question: "When was the Eiffel Tower built?"
Expected entities:
- {{"name": {"Q243" if DATASET == "wikidata" else "'Eiffel_Tower'"}, "type": "resource", "importance": 5}}
- {{"name": {"P571" if DATASET == "wikidata" else "'completionDate'"}, "type": "property", "importance": 5}}
- {{"name": {"P1619" if DATASET == "wikidata" else "'openingDate'"}, "type": "property", "importance": 3}}
- {{"name": {"Q41176" if DATASET == "wikidata" else "'Building'"}, "type": "resource", "importance": 1}}

Question: "What is the population of Tokyo?"
Expected entities:
- {{"name": {"Q1490" if DATASET == "wikidata" else "'Tokyo'"}, "type": "resource", "importance": 5}}
- {{"name": {"P1082" if DATASET == "wikidata" else "'populationTotal'"}, "type": "property", "importance": 5}}
- {{"name": {"Q515" if DATASET == "wikidata" else "'City'"}, "type": "resource", "importance": 1}}
- {{"name": {"Q3957" if DATASET == "wikidata" else "'Settlement'"}, "type": "resource", "importance": 1}}

Question: "List all books published by MIT Press"
Expected entities:
- {{"name": {"Q29133" if DATASET == "wikidata" else "'MIT_Press'"}, "type": "resource", "importance": 5}}
- {{"name": {"Q29133" if DATASET == "wikidata" else "'MIT'"}, "type": "resource", "importance": 5}}
- {{"name": {"Q571" if DATASET == "wikidata" else "'Book'"}, "type": "resource", "importance": 3}}
- {{"name": {"Q47461344" if DATASET == "wikidata" else "'WrittenWork'"}, "type": "resource", "importance": 2}}
- {{"name": {"P123" if DATASET == "wikidata" else "'publisher'"}, "type": "property", "importance": 5}}
- {{"name": {"Q17537576" if DATASET == "wikidata" else "'Work'"}, "type": "resource", "importance": 1}}

Question: "List all board games by GMT"
Expected entities:
- {{"name": {"Q5517890" if DATASET == "wikidata" else "'GMT_Games'"}, "type": "resource", "importance": 5}}
- {{"name": {"Q5517890" if DATASET == "wikidata" else "'GMT'"}, "type": "resource", "importance": 5}}
- {{"name": {"P123" if DATASET == "wikidata" else "'publisher'"}, "type": "property", "importance": 5}}
- {{"name": {"Q11019" if DATASET == "wikidata" else "'Board_Game'"}, "type": "resource", "importance": 3}}
- {{"name": {"Q11019" if DATASET == "wikidata" else "'Boardgame'"}, "type": "resource", "importance": 3}}
- {{"name": {"Q131894" if DATASET == "wikidata" else "'Wargame'"}, "type": "resource", "importance": 3}}
- {{"name": {"Q131894" if DATASET == "wikidata" else "'Tabletop_game'"}, "type": "resource", "importance": 3}}
- {{"name": {"Q11410" if DATASET == "wikidata" else "'Game'"}, "type": "resource", "importance": 1}}

Question: "{query}"
"""

    response = call_llm_api(prompt)
    text = llm_extract_response_text(response).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(text)
        if "entities" in result and result["entities"]:
            formatted_entities = []
            for entity in result["entities"]:
                if isinstance(entity, dict):
                    importance = entity.get("importance", 1)
                    if not isinstance(importance, int) or importance < 1 or importance > 5:
                        importance = 1
                    formatted_entities.append({
                        "name": entity.get("name", ""),
                        "type": entity.get("type", "resource"),
                        "importance": importance
                    })
                else:
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




def filter_keywords(keywords) -> list:
    filtered_keywords = []
    seen_keywords = set()
    for keyword in keywords:
        cleaned_keyword = (keyword or "").strip()
        if not cleaned_keyword:
            continue

        keyword_lower = cleaned_keyword.lower()

        if keyword_lower in seen_keywords:
            continue
        seen_keywords.add(keyword_lower)

        if len(cleaned_keyword.split()) == 1 and keyword_lower in CUSTOM_DROP:
            continue

        if len(cleaned_keyword.split()) == 1 and keyword_lower in ENGLISH_STOPWORDS:
            continue

        filtered_keywords.append(cleaned_keyword)
    return filtered_keywords
