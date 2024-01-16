import os
import itertools
import pandas as pd
from torch import load, save
from datasets import load_dataset
# from torchtext.data import get_tokenizer
# from transformers import AutoTokenizer
import numpy as np
import torch


def data_preprocessor():
    data = load_dataset('fashion_mnist')
    
    # Repeat for both test and train...
    for st in data.keys():
        inputs = torch.tensor(np.array(data[st]['image'])).float()
        labels = torch.tensor(np.array(data[st]['label']))

        inputs = torch.nn.functional.normalize(inputs)
        
        tensor_datset = torch.utils.data.TensorDataset(inputs, labels)

        torch.save(tensor_datset, f'data/processed/{st}.pt')


if __name__ == '__main__':
    data_preprocessor()
    # All datasets in the huggingface multi_nli library
    # das_huggingface_pirater('huggingface')
        
    # model_name = 'bert-base-uncased'
    # data_path = './data/raw/multi_nli_dataset.pt'
    # print(f'Extending tokenizer for {model_name} with words from {data_path}')
    # make_extended_vocab(data_path=data_path, model_name=model_name)
    
    # print('Processing data...')
    # dataset_preprocessor('./data/raw/multi_nli_dataset.pt', 'models/bert-base-uncased-extended-tokenizer')
