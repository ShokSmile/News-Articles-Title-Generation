# Packages
from typing import List, Dict, Tuple, Any
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import os
from tqdm import trange

SPECIAL_TOKEN = '<|TG|>'

class TitleDataset(Dataset):
    
    def __init__(self, tokenizer: AutoTokenizer, data_path: str) -> None:
        """
        TitleDataset initialization. Main assumption: our tokenizer has already all necessary special tokens.

        Args:
            tokenizer (AutoTokenizer): model tokenizer 
            data_path (str): path to existing data: train, validation or test. 

        Raises:
            ValueError: raises if there's no path to data
        """
        super().__init__()
        
        # check data data path existence
        if not os.path.exists(data_path):
            raise ValueError(f"The specified path does not exist: {data_path}. Please ensure the path is correct and points to the intended file or directory.")
        
        # parcing initial data
        data = pd.read_csv(data_path)
        self.len = len(data)
        if 'test' in data_path:
            self.text = data['text'].to_list()
            self.titles = None
            self.mode = 'test'
        else:
            self.text = data['text'].to_list()
            self.titles = data['titles'].to_list()
            self.mode = 'train/val'
        
        #tokenization
        self.tokenized_text = []
        self.tokenized_titles = []
        for i in trange(self.len, desc='Tokenization...'):
            self.tokenized_text.append(tokenizer.additional_special_tokens_ids + tokenizer(self.text[i]).input_ids)
            if self.mode != 'test':
                self.tokenized_titles.append(tokenizer(self.titles[i]).input_ids)
            
    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index) -> Any:
        if self.mode != 'test':
            return {
                "input_ids": self.tokenized_text[index],
                "labels": self.tokenized_titles[index]
            }
        else:
            return {
                "input_ids": self.tokenized_text[index],
                "labels": None
            }
        
        
if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [SPECIAL_TOKEN]}
    )
    temp = TitleDataset(tokenizer=tokenizer, data_path="../challenge_files/data/validation.csv")

    print(f"""Sample of TitleDataset: {temp[0]},
          ===========================
          Initial text: {temp.text[0]},
          ===========================
          Initial title: {temp.titles[0]}
          """)

        