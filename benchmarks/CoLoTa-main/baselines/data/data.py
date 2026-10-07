# generator functions to iterate over the datasets 

import json, os, csv


class Dataset:
    def __init__(self, dataset_name, mode):

        self.data = self._load_dataset(dataset_name, mode)
    def __call__(self, row_number):
        row = self.data[row_number]
        
        query= row[1]
        answer = row[2] 
    
        return query, answer.rstrip("\n")
    

    def _load_dataset(self, dataset_name, mode):
        """
        Loads the dataset from the file_path
        """
        # with open(os.path.join('data', dataset_name + '-' + mode + '.csv'), 'r') as f:
        #     data = f.readlines()

        
        with open(os.path.join('data', dataset_name + '-' + mode + '.csv'), mode="r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            data = list(reader)
        
        return data
    