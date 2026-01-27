# Graph Semantic Summarizer

A Python application that uses the Gemini API to extract entities and process semantic queries for graph-based knowledge summarization.

## Project Structure

```
graph-semantic-summarizer/
├── src/
│   ├── main.py                    # Main entry point for dataset processing
│   ├── config/
│   │   └── settings.py           # Configuration (API key, model settings)
│   ├── pipeline/
│   │   └── pipeline.py           # Complete pipeline orchestration
│   ├── services/
│   │   └── gemini_client.py      # Gemini API client service
│   └── utils/
│       ├── parser.py             # NLP utilities (entity extraction, response parsing)
│       ├── dbpedia.py            # DBpedia URI lookup and triple retrieval
│       ├── dataset.py            # Dataset loading and query keyword extraction
│       ├── scoring.py            # Importance score computation
│       ├── similarity.py         # Semantic similarity computation
│       ├── ranking.py            # Triple ranking and selection
│       ├── pruning.py            # URI and triple pruning
│       └── deduplication.py      # Triple deduplication
├── dataset/                      # QALD-9+ dataset files
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Files

- `src/main.py` - Main entry point for processing QALD-9+ dataset questions
- `src/pipeline/pipeline.py` - Complete pipeline orchestration with all processing steps
- `src/config/settings.py` - Configuration for API key and Gemini model settings
- `src/services/gemini_client.py` - Service for making requests to the Gemini API
- `src/utils/parser.py` - NLP utilities for entity extraction and response parsing using Gemini API
- `src/utils/dbpedia.py` - DBpedia SPARQL queries for URI lookup and triple retrieval
- `src/utils/dataset.py` - Dataset loading utilities and query keyword extraction
- `src/utils/scoring.py` - Importance score computation for entities and triples
- `src/utils/similarity.py` - Semantic similarity computation using sentence transformers
- `src/utils/ranking.py` - Final scoring, ranking, and top-K triple selection
- `src/utils/pruning.py` - URI filtering and triple cleaning utilities
- `src/utils/deduplication.py` - Triple deduplication logic
- `requirements.txt` - Python dependencies

## Versions & Libraries

### Python Version

- **Python**: 3.12.3

### Dependencies

- **requests**: >=2.31.0 (HTTP library for API calls)

### Standard Library Modules Used

- `json` - JSON encoding/decoding
- `sys` - System-specific parameters and functions
- `os` - Operating system interface (for environment variables)
- `typing` - Type hints support

### Requirements

The `requirements.txt` file specifies minimum versions:

- `requests>=2.31.0`
- `sentence-transformers>=2.2.0`
- `scikit-learn>=1.3.0`
- `numpy>=1.24.0`

## Installation

1. Install Python dependencies:

```bash
pip3 install -r requirements.txt
```

Or if pip3 is not installed, install it first:

```bash
sudo apt install python3-pip
pip3 install -r requirements.txt
```

2. Set up your Gemini API key (optional - defaults to hardcoded key):

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or modify `config/settings.py` directly.

## Usage

### Basic Usage

Run the main script to process the QALD-9+ dataset:

```bash
python src/main.py
```

This uses the default test file (`src/test/small.json`) and the dataset configured in `settings.py` or environment variable.

### Command-Line Options

You can specify the input file and other options:

```bash
# Specify input file
python src/main.py -i dataset/QALD_9_plus-main/data/qald_9_plus_test_dbpedia.json

# Specify input file and dataset (DBpedia)
python src/main.py -i dataset/QALD_9_plus-main/data/qald_9_plus_test_dbpedia.json --dataset dbpedia

# Specify input file and dataset (Wikidata)
python src/main.py -i dataset/QALD_9_plus-main/data/qald_9_plus_test_wikidata.json --dataset wikidata

# Specify custom output directory
python src/main.py -i src/test/small.json -o ./custom_output

# Use relative or absolute paths
python src/main.py -i ./src/test/small.json
python src/main.py -i /absolute/path/to/dataset.json

# Show help
python src/main.py --help
```

### Options

- `-i, --input`: Path to input JSON dataset file (default: `src/test/small.json`)
- `--dataset`: Knowledge graph dataset to use: `"dbpedia"` or `"wikidata"` (default: from `settings.py` or `DATASET` environment variable)
- `-o, --output`: Output directory for results (default: `./out`)

The script processes questions from the specified dataset file and runs each through the complete pipeline:

1. **Extract entities** using Gemini API:
   - Named entities and concepts
   - Related DBpedia categories and types
   - Implied properties and relations
   - Verbs and synonyms
2. **Lookup entity URIs** in DBpedia
3. **Retrieve triples** from DBpedia for each entity
4. **Compute importance scores** for entities and triples
5. **Prune bad URIs** and clean triples
6. **Assign importance scores** to triples based on URI importance
7. **Compute similarity scores** using sentence transformers
8. **Rank and select** top-K triples (default: top 20)
9. **Deduplicate** equivalent triples

Results are displayed for each question showing the number of triples found and the final ranked list with importance scores.

## Current Implementation Status

### Implemented Features

- ✅ Gemini API integration (`src/services/gemini_client.py`)
  - HTTP POST requests to Gemini API
  - Error handling for timeouts and network errors
  - Response status code validation
- ✅ Entity extraction using Gemini API (`src/utils/parser.py`)
  - LLM-based entity extraction - uses Gemini API to extract entities, verbs, and synonyms
  - Comprehensive entity extraction including direct entities, related concepts, and implied properties
  - DBpedia-friendly formatting with importance scoring
  - Keyword filtering with stopword removal
- ✅ Response parsing from Gemini API (`src/utils/parser.py`)
  - Extracts text from Gemini API response structure
  - Handles malformed responses gracefully
- ✅ DBpedia integration (`src/utils/dbpedia.py`)
  - SPARQL queries for entity URI lookup
  - Triple retrieval from DBpedia endpoint
  - URI extraction and mapping utilities
- ✅ Dataset processing (`src/main.py`)
  - QALD-9+ dataset loading and processing
  - Batch processing of multiple questions
  - Formatted output with results for each question
- ✅ Complete pipeline (`src/pipeline/pipeline.py`)
  - All 9 pipeline steps fully implemented
  - Entity extraction, URI lookup, triple retrieval, scoring, ranking, and deduplication

### Pipeline Steps

The complete pipeline (`src/pipeline/pipeline.py`) includes all 9 steps:

1. **STEP 1**: Pipeline initialization [x]
2. **STEP 2**: Entity extraction [x]
   - Uses Gemini API to extract entities, verbs, and synonyms
   - Extracts direct entities, related concepts, and implied properties
   - Formats entities with DBpedia-friendly naming and importance scores
3. **STEP 3**: Entity URI lookup [x]
   - Converts entity names to DBpedia URIs using SPARQL queries
   - Deduplicates entities by URI
   - Maps entity names to their corresponding URIs
4. **STEP 4 & 5**: Triple retrieval and importance computation [x]
   - Retrieves triples from DBpedia for each entity URI
   - Computes importance scores for entity URIs based on extraction importance
5. **STEP 6**: Assign importance scores to triples [x]
   - Assigns importance scores to triples based on their URI importance
   - Filters and prunes bad URIs and invalid triples
6. **STEP 7**: Compute similarity scores [x]
   - Uses sentence transformers to compute semantic similarity between query and triples
   - Embeds query and triple text for comparison
7. **STEP 8**: Final scoring and ranking [x]
   - Combines importance and similarity scores into final scores
   - Ranks triples and selects top-K (default: 20)
8. **STEP 9**: Deduplicate triples [x]
   - Removes equivalent triples that represent the same information
   - Ensures unique results in the final output

### Entity Extraction Logic

The entity extraction (`src/utils/parser.py`) uses Gemini API for comprehensive extraction:

1. **LLM-based extraction**: Uses Gemini API to extract:

   - Direct entities (importance: 3) - entities directly mentioned in the question
   - Related concepts (importance: 2) - related DBpedia categories/types
   - Implied properties/relations (importance: 2-3) - predicates implied by verbs or question patterns
   - Verbs - lemmatized verbs from the query
   - Synonyms - additional lookup variants for better DBpedia matching

2. **Entity formatting**: Formats entities with:

   - DBpedia-friendly naming (Title_Case_With_Underscores)
   - Importance scoring (1-3 scale)
   - Type classification (resource/property)

- STEP 10: Answer verbalization (currently returns dummy answer) [ ]

## Configuration

### API Settings

Edit `src/config/settings.py` to configure:

- **API_KEY**: Your Gemini API key (defaults to environment variable `GEMINI_API_KEY`, with fallback to hardcoded key)
- **MODEL_NAME**: Gemini model to use (default: `gemini-2.5-flash-lite`)
- **BASE_URL**: Gemini API base URL (default: `https://generativelanguage.googleapis.com/v1beta`)
- **API_URL**: Full API endpoint URL (constructed from BASE_URL and MODEL_NAME)

### Model Information

- **Current model**: `gemini-2.5-flash-lite`
- **API endpoint**: `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent`
- **Request method**: POST
- **Timeout**: 30 seconds (configurable in `call_gemini_api` function)
- **Request format**: JSON with `contents` array containing `parts` with `text`
- **DBpedia endpoint**: `https://dbpedia.org/sparql` (used for URI lookup and triple retrieval)

## Git Status

### Current Branch

- **Branch**: `main`
- **Status**: Up to date with `origin/main`

### Modified Files (Not Staged)

The following files have been modified but not yet committed:

- `README.md` - Documentation updates
- `src/config/settings.py` - Configuration updates (API key, model settings, URL structure)
- `src/main.py` - Dataset processing implementation
- `src/pipeline/pipeline.py` - Complete pipeline with all 9 steps implemented
- `requirements.txt` - Dependency updates (requests, sentence-transformers, scikit-learn, numpy)
- `src/utils/parser.py` - Enhanced entity extraction with LLM refinement logic
- `src/utils/dbpedia.py` - DBpedia SPARQL integration for URI lookup and triple retrieval
- `src/utils/scoring.py` - Importance score computation
- `src/utils/similarity.py` - Semantic similarity computation
- `src/utils/ranking.py` - Triple ranking and selection
- `src/utils/pruning.py` - URI and triple pruning utilities
- `src/utils/deduplication.py` - Triple deduplication logic
- `src/utils/dataset.py` - Dataset loading and keyword extraction

### Tracked Files

All project files are tracked in git:

- `.gitignore` - Git ignore rules
- `README.md` - Project documentation
- `src/config/settings.py` - Configuration settings
- `src/main.py` - Main entry point
- `src/pipeline/pipeline.py` - Pipeline orchestration
- `requirements.txt` - Python dependencies
- `src/services/gemini_client.py` - Gemini API client service
- `src/utils/parser.py` - NLP utilities
- `src/utils/dbpedia.py` - DBpedia integration
- `src/utils/dataset.py` - Dataset utilities
- `src/utils/scoring.py` - Scoring utilities
- `src/utils/similarity.py` - Similarity computation
- `src/utils/ranking.py` - Ranking utilities
- `src/utils/pruning.py` - Pruning utilities
- `src/utils/deduplication.py` - Deduplication utilities

### Ignored Files

The following are ignored by git (see `.gitignore`):

- `__pycache__/` - Python bytecode cache directories
- `*.pyc`, `*.pyo` - Compiled Python files
- `.env`, `venv/`, `env/` - Environment and virtual environment files
- IDE-specific files (`.vscode/`, `.idea/`, `*.swp`, etc.)
- Log files (`*.log`)
- Distribution and build artifacts

## Implementation Details

### Architecture

The application follows a modular architecture:

1. **Configuration Layer** (`src/config/settings.py`)

   - Centralized configuration management
   - Environment variable support with fallback values
   - API endpoint construction

2. **Service Layer** (`src/services/gemini_client.py`)

   - Encapsulates Gemini API communication
   - Handles HTTP requests, error handling, and timeouts
   - Returns structured JSON responses

3. **Utility Layer** (`src/utils/`)

   - **parser.py**: LLM-based entity extraction using Gemini API, response parsing, keyword filtering
   - **dbpedia.py**: DBpedia SPARQL queries for URI lookup and triple retrieval
   - **dataset.py**: Dataset loading and query keyword extraction
   - **scoring.py**: Importance score computation for entities and triples
   - **similarity.py**: Semantic similarity computation using sentence transformers
   - **ranking.py**: Final scoring, ranking, and top-K selection
   - **pruning.py**: URI filtering and triple cleaning
   - **deduplication.py**: Triple deduplication logic

4. **Pipeline Layer** (`src/pipeline/pipeline.py`)

   - Orchestrates all 9 pipeline steps
   - Coordinates entity extraction, URI lookup, triple retrieval, scoring, ranking, and deduplication
   - Returns ranked triples with importance scores

5. **Application Layer** (`src/main.py`)
   - Processes QALD-9+ dataset questions
   - Handles batch processing and output formatting
   - Displays results for each processed question

### Entity Extraction Strategy

The entity extraction uses a comprehensive LLM-based approach:

1. **Direct entities** - Extracts all entities directly mentioned in the question with high importance (3)
2. **Related concepts** - Includes related DBpedia categories/types with medium importance (2)
3. **Implied properties** - Extracts predicates/properties implied by verbs or question patterns (importance 2-3)
4. **Verb extraction** - Extracts and lemmatizes verbs from the query
5. **Synonym expansion** - Includes alternative spellings and lookup variants for better DBpedia matching

This ensures robust entity extraction even for ambiguous queries by leveraging the Gemini API's understanding capabilities.

## Notes

- Make sure your API key is correct in `src/config/settings.py` or set as `GEMINI_API_KEY` environment variable
- The API endpoint uses `gemini-2.5-flash-lite` model by default
- Timeout is set to 30 seconds by default
- You can modify the model name in `src/config/settings.py` if needed
- All entity extraction is handled by the Gemini API, ensuring comprehensive and accurate extraction
- The pipeline processes questions from `src/test/small.json` by default
- DBpedia SPARQL endpoint is used for URI lookup and triple retrieval
- Sentence transformers are used for semantic similarity computation (requires model download on first run)
- Top-K selection defaults to 20 triples per question
