# main file to run experiments 
from data import data
import os, sys, copy
import tqdm, json
import numpy as np
import pickle
import argparse, time
from agent import agent
from utils.logger import logger


def get_args():
    parser = argparse.ArgumentParser(description="Run experiments for the dataset")
    parser.add_argument('path', help='Path to directory containing dataset')
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--scoring_method", type=str, default="typed resolution", help="Method to use for scoring the options")
    parser.add_argument("--experiment_name", type=str, required=True, default="res", help="Name for saving the results of the experiment")
    parser.add_argument("--llm_name", type=str, default="gemini", help="Name of the LLM model to use")
    parser.add_argument("--mode", type=str, default="original", help="original or modified")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    dataset_name = args.dataset_name
    path = args.path
    scoring_method = args.scoring_method
    llm_name = args.llm_name
    mode = args.mode

    if scoring_method == "zero shot CoT":
        Reasoner = agent.ZSCoT()
    elif scoring_method == "few shot CoT":
        Reasoner = agent.FSCoT()

    else:
        raise ValueError("Scoring method not supported")



    log = logger(args.experiment_name)
    log("Dataset: ", dataset_name); log("Path: ", path); log("Scoring Method: ", scoring_method); log("LLM Name: ", llm_name); log("Mode: ", mode)
    # load the dataset
    dataset = data.Dataset(dataset_name, mode)
    outcomes = []

    

    for i in tqdm.tqdm(range(1,151)):

        time.sleep(5)

        log("Example no: ", i)

        query,  answer= dataset(i)


        print(query)
        log("Query: ", query); log("Answer: ", answer)


        selected_option = Reasoner(query, llm_name, dataset_name, log)
        
        log("Selected option:", selected_option)

        correctness = str(answer).lower() in str(selected_option).lower()
        outcomes.append(correctness)
        if not correctness:
            log("Incorrect Answer", answer)

        print("Accuracy so far: ", np.mean(outcomes))

    
    print(f"Accuracy: {np.mean(outcomes)}"); print(f"Correct: {np.sum(outcomes)}"); print(f"Incorrect: {np.sum(np.logical_not(outcomes))}")
    log("Accuracy: ", np.mean(outcomes)); log("Correct: ", np.sum(outcomes)); log("Incorrect: ", np.sum(np.logical_not(outcomes)))
    

