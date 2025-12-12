# Graph Semantic Summarizer

A Python application that uses the Gemini API and spaCy NLP to extract entities and process semantic queries for graph-based knowledge summarization.

## Project Structure

```
graph-semantic-summarizer/
├── main.py                    # Main entry point and pipeline orchestration
├── config/
│   └── settings.py           # Configuration (API key, model settings)
├── services/
│   └── gemini_client.py      # Gemini API client service
├── utils/
│   └── parser.py             # NLP utilities (entity extraction, response parsing)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Files

- `main.py` - Main entry point with pipeline orchestration and interactive CLI
- `config/settings.py` - Configuration for API key and Gemini model settings
- `services/gemini_client.py` - Service for making requests to the Gemini API
- `utils/parser.py` - NLP utilities using spaCy for entity extraction and response parsing
- `requirements.txt` - Python dependencies

## Versions & Libraries

### Python Version

- **Python**: 3.12.3

### Dependencies

- **requests**: >=2.31.0 (HTTP library for API calls)
- **spacy**: >=3.7.0 (Natural Language Processing library)

### Standard Library Modules Used

- `json` - JSON encoding/decoding
- `sys` - System-specific parameters and functions
- `os` - Operating system interface (for environment variables)
- `typing` - Type hints support

### Requirements

The `requirements.txt` file specifies minimum versions:

- `requests>=2.31.0`
- `spacy>=3.7.0`

**Note**: After installing spacy, you need to download the English language model:

```bash
python -m spacy download en_core_web_sm
```

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

2. Download the spaCy English language model:

```bash
python -m spacy download en_core_web_sm
```

3. Set up your Gemini API key (optional - defaults to hardcoded key):

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or modify `config/settings.py` directly.

## Usage

Run the main script interactively:

```bash
python3 main.py
```

The script will prompt you to enter a query. The pipeline will:

1. **Extract entities** using spaCy NLP:
   - Named entities (NER)
   - Proper nouns
   - Subject nouns
   - Common nouns (fallback)
   - Verbs (lemmatized)
2. **Conditionally refine entities** using Gemini API if extraction is weak
3. **Call the Gemini API** with your prompt
4. **Extract and display** the response text
5. **Process through pipeline** (currently returns dummy answer)

**Note**: A query is required. The script will exit if no query is provided.

## Current Implementation Status

### Implemented Features

- ✅ Gemini API integration (`services/gemini_client.py`)
  - HTTP POST requests to Gemini API
  - Error handling for timeouts and network errors
  - Response status code validation
- ✅ Entity extraction using spaCy (`utils/parser.py`)
  - Named Entity Recognition (NER) - extracts named entities from text
  - Proper noun extraction (PROPN) - identifies proper nouns
  - Subject noun extraction (nsubj, nsubjpass) - extracts grammatical subjects
  - Common noun fallback - extracts any nouns if other methods fail
  - Verb extraction - extracts verbs and lemmatizes them
  - LLM refinement logic - conditionally uses Gemini API to refine entities when spaCy extraction is weak
- ✅ Response parsing from Gemini API (`utils/parser.py`)
  - Extracts text from Gemini API response structure
  - Handles malformed responses gracefully
- ✅ Interactive CLI interface (`main.py`)
  - User-friendly prompts and formatted output
  - Error handling and validation
- ✅ Pipeline structure (`main.py`)
  - Modular pipeline function for processing queries
  - Currently implements entity extraction and Gemini API call

### Pipeline Steps

The current pipeline (`main.py`) includes:

1. **STEP 1**: Pipeline initialization [x]
2. **STEP 2**: Entity extraction [x]
   - Uses spaCy to extract entities (NER, proper nouns, subjects, common nouns)
   - Extracts verbs from the query
   - Conditionally uses LLM refinement if extraction is weak

### Entity Extraction Logic

The entity extraction (`utils/parser.py`) implements a smart fallback strategy:

1. **Primary extraction**: Uses spaCy to extract:

   - Named entities (NER)
   - Proper nouns
   - Subject nouns
   - Common nouns (fallback)

2. **LLM refinement**: Conditionally calls Gemini API when:

   - No entities found (only "UnknownEntity")
   - Only generic nouns found (e.g., "father", "mother", "city")
   - No verbs detected
   - Query contains pronouns requiring interpretation

3. **Verb extraction**: Extracts and lemmatizes all verbs from the query

### Future Pipeline Steps (Commented Out)

The following steps are planned but not yet implemented:

- STEP 3: Entity URI lookup [ ]
- STEP 4: Triple retrieval [ ]
- STEP 5: Importance score computation [ ]
- STEP 6: Similarity score computation [ ]
- STEP 7: Scoring and ranking triples [ ]
- STEP 8: Top K selection [ ]
- STEP 9: Answer verbalization (currently returns dummy answer) [ ]

## Configuration

### API Settings

Edit `config/settings.py` to configure:

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

## Git Status

### Current Branch

- **Branch**: `main`
- **Status**: Up to date with `origin/main`

### Modified Files (Not Staged)

The following files have been modified but not yet committed:

- `README.md` - Documentation updates
- `config/settings.py` - Configuration updates (API key, model settings, URL structure)
- `main.py` - Pipeline implementation with entity extraction and Gemini API integration
- `requirements.txt` - Dependency updates (requests, spacy)
- `utils/parser.py` - Enhanced entity extraction with LLM refinement logic

### Tracked Files

All project files are tracked in git:

- `.gitignore` - Git ignore rules
- `README.md` - Project documentation
- `config/settings.py` - Configuration settings
- `main.py` - Main entry point
- `requirements.txt` - Python dependencies
- `services/gemini_client.py` - Gemini API client service
- `utils/parser.py` - NLP utilities

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

1. **Configuration Layer** (`config/settings.py`)

   - Centralized configuration management
   - Environment variable support with fallback values
   - API endpoint construction

2. **Service Layer** (`services/gemini_client.py`)

   - Encapsulates Gemini API communication
   - Handles HTTP requests, error handling, and timeouts
   - Returns structured JSON responses

3. **Utility Layer** (`utils/parser.py`)

   - NLP processing with spaCy
   - Entity extraction with multiple strategies
   - Response parsing from API
   - LLM refinement logic

4. **Application Layer** (`main.py`)
   - Orchestrates the pipeline
   - Provides CLI interface
   - Handles user input and output formatting

### Entity Extraction Strategy

The entity extraction uses a multi-tier approach:

1. **Tier 1**: Named Entity Recognition (NER) - spaCy's built-in entity recognition
2. **Tier 2**: Proper nouns - Identifies capitalized proper nouns
3. **Tier 3**: Subject nouns - Extracts grammatical subjects
4. **Tier 4**: Common nouns - Fallback to any nouns
5. **Tier 5**: LLM refinement - Uses Gemini API when extraction is insufficient

This ensures robust entity extraction even for ambiguous queries.

## Notes

- Make sure your API key is correct in `config/settings.py` or set as `GEMINI_API_KEY` environment variable
- The API endpoint uses `gemini-2.5-flash-lite` model by default
- Timeout is set to 30 seconds by default
- You can modify the model name in `config/settings.py` if needed
- The spaCy model `en_core_web_sm` must be downloaded before first use
- The application will automatically use LLM refinement when entity extraction is weak
