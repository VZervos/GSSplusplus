from abc import ABC, abstractmethod 
from sentence_transformers import SentenceTransformer, util
from utils.logic.fol import *
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import openai
from utils.embeddings_utils import get_embedding
import pickle

load_dotenv()
client = OpenAI()


class Retriever(ABC):

    def __init__(self):
        pass 
    @abstractmethod
    def __call__(self): 
        pass 

    def query2nl(self, query):
        if isinstance(query, Predicate):
            return f"x is a {query.name}"
        elif isinstance(query, Not):
            return f"x is not a {query.arg.name}"
    
    def rules2nl(self, rules):
        nl2rules = {}
        rules_nl = []
        for rule in rules:
            nl = str(rule)
            nl2rules[nl] = rule
            rules_nl.append(nl)
            # if isinstance(rule, ForAll):
            #     if isinstance(rule.body.rhs, Predicate):
            #         nl = f"Every {rule.body.lhs.name} is a {rule.body.rhs.name}"
            #         nl2rules[nl] = rule
            #         rules_nl.append(nl)
            #     elif isinstance(rule.body.rhs, Not):
            #         nl = f"Every {rule.body.lhs.name} is not a {rule.body.rhs.arg.name}"
            #         nl2rules[nl] = rule
            #         rules_nl.append(nl)
        return rules_nl, nl2rules
    def encode(self, s):
        pass

class GPT3_Retriever(Retriever):
    def __init__(self, dataset_name ,model_dir):
        self.dataset_name = dataset_name
        try:
            with open(f'data/{dataset_name}_tbox_nl2rules.pkl', "rb") as f:
                self.nl2rules_tbox = pickle.load(f)
            with open(f'data/{dataset_name}_tbox_rules_nl.pkl', "rb") as f:
                self.rules_nl_tbox = pickle.load(f)
        except:
            self.nl2rules_tbox = None
        
        try:
            with open(f'data/{dataset_name}_abox_nl2rules.pkl', "rb") as f:
                self.nl2rules_abox = pickle.load(f)
            with open(f'data/{dataset_name}_abox_rules_nl.pkl', "rb") as f:
                self.rules_nl_abox = pickle.load(f)
        except:
            self.nl2rules_abox = None
    

    def __call__(self, query, rules, k_facts=10, assumption=None):

        t_box = []
        for rule in rules:
            if isinstance(rule, ForAll):
                t_box.append(rule)
        a_box_rules = [rule for rule in rules if rule not in t_box]

        if self.nl2rules_tbox:
            nl2rules_tbox = self.nl2rules_tbox
            rules_nl_tbox = self.rules_nl_tbox
        else:
            rules_nl_tbox, nl2rules_tbox = self.rules2nl(t_box)
        
        if self.nl2rules_abox:
            nl2rules_abox = self.nl2rules_abox
            rules_nl_abox = self.rules_nl_abox
        else:
            rules_nl_abox, nl2rules_abox = self.rules2nl(a_box_rules)
        # rules = a_box_rules
        query_nl = str(query)
        if assumption:
            assumption_nl = str(assumption)
            query_nl = f"{query_nl} and {assumption_nl}"
        else:
            query_nl = query_nl

        query_embedding = self.encode(query_nl)
        if self.nl2rules_tbox:
            with open(f"data/{self.dataset_name}_tbox_embeddings_gpt3.pkl", "rb") as f:
                rules_embeddings_tbox = pickle.load(f)
        else:
            rules_embeddings_tbox = self.encode(rules_nl_tbox)
        if self.nl2rules_abox:
            with open(f"data/{self.dataset_name}_abox_embeddings_gpt3.pkl", "rb") as f:
                rules_embeddings_abox = pickle.load(f)
        else:
            rules_embeddings_abox = self.encode(rules_nl_abox)

        results_tbox = self.select_most_similar(query_embedding, rules_embeddings_tbox, k_facts)
        results_abox = self.select_most_similar(query_embedding, rules_embeddings_abox, k_facts)
        if assumption:
            selected_rules = [nl2rules_abox[rules_nl_abox[result]] for result in results_abox] + [assumption]
            rules.append(assumption)
        else:
            selected_rules = [nl2rules_abox[rules_nl_abox[result]] for result in results_abox]
        selected_rules += [nl2rules_tbox[rules_nl_tbox[result]] for result in results_tbox]
        
        return selected_rules
    
    def encode(self, s):
        return client.embeddings.create(input=s, model= "text-embedding-3-small").data

    def select_most_similar(self, query_embedding, rules_embeddings, k_facts):
        rule_scores = {}
        query_vector = query_embedding[0].embedding
        for i, rule_embedding in enumerate(rules_embeddings):
            rule_vector = rule_embedding.embedding
            rule_scores[i] = self.cosine_similarity(query_vector, rule_vector)
        top_k = sorted(rule_scores, key=rule_scores.get, reverse=True)[:k_facts]
        return top_k
    
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class ST_Retriever(Retriever):
    def __init__(self, dataset_name, model_dir):
        self.dataset_name = dataset_name
        self.mpnet = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            cache_folder=f"{model_dir}/sentence_transformers",
        )
        try:
            with open(f"data/{self.dataset_name}_tbox_nl2rules.pkl", "rb") as f:
                self.nl2rules_tbox = pickle.load(f)
            with open(f"data/{self.dataset_name}_tbox_rules_nl.pkl", "rb") as f:
                self.rules_nl_tbox = pickle.load(f)
        except:
            self.nl2rules_tbox = None

        try:
            with open(f"data/{self.dataset_name}_abox_nl2rules.pkl", "rb") as f:
                self.nl2rules_abox = pickle.load(f)
            with open(f"data/{self.dataset_name}_abox_rules_nl.pkl", "rb") as f:
                self.rules_nl_abox = pickle.load(f)
        except:
            self.nl2rules_abox = None

    def __call__(self, query, rules, k_facts=10, assumption=None):
        t_box = []
        for rule in rules:
            if isinstance(rule, ForAll):
                t_box.append(rule)
        a_box_rules = [rule for rule in rules if rule not in t_box]

        if self.nl2rules_tbox:
            nl2rules_tbox = self.nl2rules_tbox
            rules_nl_tbox = self.rules_nl_tbox
        else:
            rules_nl_tbox, nl2rules_tbox = self.rules2nl(t_box)
        
        if self.nl2rules_abox:
            nl2rules_abox = self.nl2rules_abox
            rules_nl_abox = self.rules_nl_abox
        else:
            rules_nl_abox, nl2rules_abox = self.rules2nl(a_box_rules)
        query_nl = str(query)
        if assumption:
            assumption_nl = str(assumption)
            query_nl = f"{query_nl} and {assumption_nl}"
        else:
            query_nl = query_nl
        query_embedding = self.encode([query_nl])
        if self.nl2rules_tbox:
            with open(f"data/{self.dataset_name}_tbox_embeddings_st.pkl", "rb") as f:
                rules_embeddings_tbox = pickle.load(f)
        else:
            rules_embeddings_tbox = self.encode(rules_nl_tbox)

        if self.nl2rules_abox:
            with open(f"data/{self.dataset_name}_abox_embeddings_st.pkl", "rb") as f:
                rules_embeddings_abox = pickle.load(f)
        else:
            rules_embeddings_abox = self.encode(rules_nl_abox)
        

        results_abox = util.semantic_search(
            query_embedding, rules_embeddings_abox, score_function=util.dot_score, top_k=k_facts
        )[0]
        # results_tbox = util.semantic_search(
        #     query_embedding, rules_embeddings_tbox, score_function=util.dot_score, top_k=k_facts
        # )[0]
        if assumption:
            selected_rules = [nl2rules_abox[rules_nl_abox[result["corpus_id"]]] for result in results_abox] + [assumption]
        else:
            selected_rules = [nl2rules_abox[rules_nl_abox[result["corpus_id"]]] for result in results_abox]
        
        # selected_rules += [nl2rules_tbox[rules_nl_tbox[result["corpus_id"]]] for result in results_tbox]
        selected_rules += [rule for rule in t_box]
        return selected_rules


    def encode(self, s):
        return self.mpnet.encode(s, normalize_embeddings=True, convert_to_tensor=True)