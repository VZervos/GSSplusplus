import os

# Your API key
API_KEY = os.getenv("GEMINI_API_KEY", "ADD IT HERE !!!!!!!!!!!!!!!!!!!!!!!!!")

# API endpoint for Gemini 2.5 Flash change model name to use different models
MODEL_NAME = "gemini-2.5-flash"

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
API_URL = f"{BASE_URL}/models/{MODEL_NAME}:generateContent"
