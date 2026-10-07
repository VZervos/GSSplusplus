# main file for our agent
import os, sys, json, re, random, copy, signal
import multiprocessing
random.seed(98)
from abc import ABC, abstractmethod 
import numpy as np
import time
from agent.llm.llm_actions import *


class Agent(ABC):
    def __init__(self):
        pass 
    @abstractmethod
    def __call__(self):
        pass 


    @abstractmethod
    def _prove(self):
        pass 



class ZSCoT(Agent):
    def __init__(self):
        super().__init__()


    def __call__(self, query, llm_name, dataset_name, log):
        answer = self._prove(query, llm_name, dataset_name, log=log)
        return str(answer)
    
    def _prove(self, query, llm_name, dataset_name, log=None):
        answer, proof = GetMonolithicProof(llm_name, dataset_name, 'agent/llm/llm_prompts')(QUERY=query)
        log("proof", proof)
        return answer

class FSCoT(Agent):
    def __init__(self):
        super().__init__()

    def __call__(self, query, llm_name, dataset_name, log):
        answer = self._prove(query, llm_name, dataset_name, log=log)
        return str(answer)
    
    def _prove(self, query, llm_name, dataset_name, log=None):

        answer, proof = GetFewshotProof(llm_name, dataset_name, 'agent/llm/llm_prompts')(QUERY=query)
        log("proof", proof)
        return answer
