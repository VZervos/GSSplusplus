# Gemini API Test - Python Version

A simple command-line Python script to test the Gemini API.

## Files

- `gemini_test.py` - Command-line script to interact with Gemini API
- `requirements.txt` - Python dependencies

## Versions & Libraries

### Python Version

- **Python**: 3.12.3

### Dependencies

- **requests**: 2.31.0 (HTTP library for API calls)

### Standard Library Modules Used

- `json` - JSON encoding/decoding
- `sys` - System-specific parameters and functions
- `typing` - Type hints support

### Requirements

The `requirements.txt` file specifies minimum versions:

- `requests>=2.31.0`

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

## Usage

Run the script with your prompt:

```bash
python3 gemini_test.py "Your prompt here"
```

**Note**: A prompt is required. The script will exit if no prompt is provided.

## Why Python?

- ✅ **No CORS issues** - Server-side requests bypass browser CORS restrictions
- ✅ **Simple to test** - Just run the script with a prompt
- ✅ **Easy to debug** - See full error messages in terminal
- ✅ **More reliable** - No browser security restrictions

## Notes

- Make sure your API key is correct in the script
- The API endpoint uses `gemini-2.5-flash` model
- Timeout is set to 30 seconds by default
- You can modify the model name in the script if needed
