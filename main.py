import sys
from services.gemini_client import call_gemini_api
from utils.parser import extract_response_text


def main():
    if len(sys.argv) <= 1:
        print("💡 No prompt provided. Usage: python main.py \"prompt here\"")
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])

    try:
        response = call_gemini_api(prompt)
        text = extract_response_text(response)

        print("=" * 60)
        print("🤖 GEMINI RESPONSE:")
        print("=" * 60)
        print(text)
        print("=" * 60)

    except Exception as e:
        print("=" * 60)
        print("❌ ERROR:")
        print("=" * 60)
        print(str(e))
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
