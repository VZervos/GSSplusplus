import os
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
import google.generativeai as genai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

load_dotenv()
#GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
GOOGLE_API_KEY= ''
#GROQ_API_KEY=os.getenv('GROQ_API_KEY')
GROQ_API_KEY= ''

genai.configure(api_key=GOOGLE_API_KEY)
client = OpenAI()

class LLM:

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def __call__(self, messages, model="gemini"):
        if "gpt" in model.lower():
            model = "o1-preview"
            if "o1" in model.lower():
                messages[1]['content'] = messages[0]['content'] + " " + messages[1]['content']
                messages = messages[1:]
            client = OpenAI(api_key='')
            completion = client.chat.completions.create(
            model=model, messages=messages, max_completion_tokens=3000)
            return completion.choices[0].message.content 
        
        elif "gemini" in model.lower():
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(messages)
            return response.text
        
        elif "groq" not in model.lower():
            prompt = messages[0] + " " + messages[1]
            client = OpenAI(base_url="https://gpuXXX:XXXX/v1", api_key="EMPTY")
            completion = client.completions.create(
                model = "/model-weights/llama-3.3-70b-versatile",
                prompt=prompt
            )
            return completion.choices[0].text

        elif "groq" in model.lower():
            client = Groq(api_key=GROQ_API_KEY)
            completion = client.chat.completions.create(
                model= "llama-3.3-70b-versatile", messages=messages)
            return completion.choices[0].message.content
