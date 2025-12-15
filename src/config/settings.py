import os

# Your API key
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDWd89_G9SAs95nAaMlfS0_Xt4ADelJ9TQ")

# API endpoint for Gemini 2.5 Flash change model name to use different models
MODEL_NAME = "gemini-2.5-flash-lite"

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
API_URL = f"{BASE_URL}/models/{MODEL_NAME}:generateContent"
