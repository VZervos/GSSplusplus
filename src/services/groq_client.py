from groq import Groq
from config.settings import GROQ_API_KEY, GROQ_MODEL

def call_groq_api(prompt: str, api_key: str = GROQ_API_KEY, timeout: int = 30) -> dict:

    if not api_key:
        raise Exception(
            "GROQ_API_KEY is not set. Please set it via environment variable:\n"
            "  export GROQ_API_KEY='your_api_key_here'\n"
            "Or get a free API key at: https://console.groq.com/keys"
        )
    
    print(f"Sending request to Groq API (model: {GROQ_MODEL})...")
    print(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else prompt)
    
    try:
        client = Groq(api_key=api_key, timeout=timeout)
        
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,  # Deterministic for consistency
            max_tokens=2048,
            top_p=1,
            stream=False
        )
        
        print(f"Response received from Groq API")
        
        response_text = completion.choices[0].message.content
        gemini_compatible_response = {
            "candidates": [{
                "content": {
                    "parts": [{"text": response_text}]
                }
            }]
        }
       
        return gemini_compatible_response
    
    except Exception as e:
        error_msg = str(e)
        
        if "rate_limit" in error_msg.lower():
            raise Exception(f"Groq API rate limit exceeded: {error_msg}")
        elif "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            raise Exception(f"Groq API authentication failed: {error_msg}")
        elif "timeout" in error_msg.lower():
            raise Exception(f"Groq API request timeout after {timeout} seconds")
        else:
            raise Exception(f"Groq API error: {error_msg}")

