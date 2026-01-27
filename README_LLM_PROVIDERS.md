# LLM Provider Configuration Guide

This project now supports **two LLM providers** for entity extraction:

1. **Gemini** (Google) - Original provider
2. **Groq** (llama-3.1-8b-instant) - New FREE provider with ~30 RPM, >1000 requests/day

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install just the Groq SDK:

```bash
pip install groq>=0.4.0
```

### 2. Get API Keys

**Gemini API Key:**
- Visit: https://makersuite.google.com/app/apikey
- Create or use existing API key

**Groq API Key (FREE):**
- Visit: https://console.groq.com/
- Sign up for free account
- Create API key from dashboard

### 3. Set Environment Variables

**To use Gemini (default):**
```bash
export GEMINI_API_KEY="your_gemini_api_key"
# LLM_PROVIDER defaults to "gemini" if not set
```

**To use Groq:**
```bash
export GROQ_API_KEY="your_groq_api_key"
export LLM_PROVIDER="groq"
```

**To switch between providers:**
```bash
# Use Gemini
export LLM_PROVIDER="gemini"

# Use Groq
export LLM_PROVIDER="groq"
```

### 4. Run the Pipeline

```bash
# Run with default provider (Gemini)
python3 src/main.py

# Or explicitly set provider
LLM_PROVIDER=groq python3 src/main.py
```

## Testing

### Verify Static Implementation

Run the static verification script (no API calls needed):

```bash
python3 verify_implementation.py
```

This checks that all files are correctly implemented.

### Test Both Providers

Run the integration test script:

```bash
python3 test_llm_providers.py
```

This will:
- Test both Gemini and Groq (if API keys are set)
- Validate JSON output format
- Check entity structure
- Verify importance scores (1-5 range)

### Test with Sample Query

```bash
# Test Gemini
export LLM_PROVIDER="gemini"
export GEMINI_API_KEY="your_key"
python3 src/main.py

# Test Groq
export LLM_PROVIDER="groq"
export GROQ_API_KEY="your_key"
python3 src/main.py
```

## Provider Comparison

| Feature | Gemini | Groq |
|---------|--------|------|
| **Model** | gemini-2.5-flash-lite | llama-3.1-8b-instant |
| **Cost** | Paid (generous free tier) | FREE |
| **Rate Limit** | Variable | ~30 RPM |
| **Daily Limit** | Variable | >1000 requests |
| **Speed** | Fast | Very Fast (~560 tokens/sec) |
| **Best For** | Production, high volume | Development, testing, cost-sensitive |


## Architecture

```
Query → pipeline.py
         ↓
      parser.py (extract_entities)
         ↓
      llm_client.py (dispatcher)
         ↓
    ┌────────────────┐
    ↓                ↓
gemini_client.py  groq_client.py
    ↓                ↓
    └────────────────┘
         ↓
   Unified Response Format
         ↓
      parser.py (JSON parsing)
         ↓
   Normalized Entities
```


