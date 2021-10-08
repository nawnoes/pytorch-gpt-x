import warnings
warnings.filterwarnings("ignore")

import os
import logging
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from arg import ModelConfig
from transformers import BertTokenizer

class GPT3Dataset(Dataset):
    def __init__(self, tokenizer, max_len, dir_path):
        logging.info('Start pretraining data load!')

        self.tokenizer = tokenizer
        self.max_len =max_len
        self.docs = []

        # 파일 리스트
        file_list = os.listdir(dir_path)

        # num_lines = sum(1 for line in open(path, 'r',encoding='utf-8'))
        file_progress_bar = tqdm(file_list, position=0, leave=True, bar_format='{l_bar}{bar:10}{r_bar}')
        for file_name in file_progress_bar:
            path = f'{dir_path}/{file_name}'
            data_file =  open(path, 'r',encoding='utf-8')
            for line in tqdm(data_file,
                             desc='Data load for pretraining',
                             position=1, leave=True):
                line = line[:-1]
                self.docs.append(line)
        logging.info('Complete data load')

    def _tokenize_input_ids(self, input_ids: list, add_special_tokens:bool = False, pad_to_max_length: bool = True):
        inputs = torch.tensor(self.tokenizer.encode(input_ids, add_special_tokens=add_special_tokens, max_length=self.max_len, pad_to_max_length=pad_to_max_length, return_tensors='pt',truncation=True))
        return inputs
    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        inputs = self._tokenize_input_ids(self.docs[idx], pad_to_max_length=True)
        labels = inputs.clone()

        inputs= inputs.squeeze()
        labels= labels.squeeze()


        return inputs, labels

class GPTXDataset(Dataset):
    def __init__(self, tokenizer, max_len, dir_path):
        logging.info('Start pretraining data load!')

        self.tokenizer = tokenizer
        self.max_len =max_len
        self.docs = []

        # 파일 리스트
        file_list = os.listdir(dir_path)

        # num_lines = sum(1 for line in open(path, 'r',encoding='utf-8'))
        file_progress_bar = tqdm(file_list, position=0, leave=True, bar_format='{l_bar}{bar:10}{r_bar}')
        for file_name in file_progress_bar:
            path = f'{dir_path}/{file_name}'
            data_file =  open(path, 'r',encoding='utf-8')

            tmp_line = [self.tokenizer.cls_token_id]
            for line in tqdm(data_file,
                             desc='Data load for pretraining',
                             position=1, leave=True):
                line = line[:-1]
                line_ids = self.tokenizer.encode(line, add_special_tokens=False, pad_to_max_length=False,
                                                 max_length=max_len - 2, truncation=True)
                line_ids += [self.tokenizer.sep_token_id]

                if len(tmp_line) + len(line_ids) < self.max_len:
                    tmp_line += line_ids
                else:
                    self.docs.append(tmp_line)
                    tmp_line = [self.tokenizer.cls_token_id]

        logging.info('Complete data load')

    def _tokenize_input_ids(self, input_ids: list, add_special_tokens:bool = False, pad_to_max_length: bool = True):
        inputs = torch.tensor(self.tokenizer.encode(input_ids, add_special_tokens=add_special_tokens, max_length=self.max_len, pad_to_max_length=pad_to_max_length, return_tensors='pt',truncation=True))
        return inputs
    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        inputs = self._tokenize_input_ids(self.docs[idx], pad_to_max_length=True)
        labels = inputs.clone()

        inputs= inputs.squeeze()
        labels= labels.squeeze()


        return inputs, labels

if __name__=='__main__':
    data_path = './data/train/sample.txt'
    config_path = './config.json'
    config = ModelConfig(config_path=config_path).get_config()

    # Tokenizer
    tokenizer = BertTokenizer(vocab_file=config.vocab_path, do_lower_case=False)
    dataset = GPTXDataset(tokenizer,config.max_seq_len, config.data_path)
    print(dataset)
