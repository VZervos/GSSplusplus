from abc import ABC, abstractmethod

from utils.llm.llm import LLM
from utils.llm.prompt import Prompt


class Chain(ABC):
    def __init__(self, prompt_f, llm_name):
        self.llm_name = llm_name
        self.prompt = Prompt(prompt_f, llm_name)
        self.llm = LLM()

    def __call__(self, **prompt_input):
        if "gpt" in self.llm_name.lower():
            prompt_input = {k.upper(): v for k, v in prompt_input.items()}
            messages = self.prompt(**prompt_input) 
        elif "gemini" in self.llm_name.lower() or "groq" in self.llm_name.lower():
            messages = self.prompt(**prompt_input)
        response = self.llm(messages, self.llm_name)
        result = self.parse_response(response)

        #return messages, response, result
        return result
    
    @abstractmethod
    def parse_response(self, response):
        """Parse the LLM's response.
        
        Args:
            response (str): the LLM's response.
        
        Returns:
            The parsed response.
        """
