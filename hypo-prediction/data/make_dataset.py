import pandas as pd
import torch
from datasets import load_dataset
import os
import torchtext


def das_huggingface_pirater():
    """Pirates the huggingface library
    Yeah no it doesn't really pirate it, but you know"""

    print('Loading multi_nli dataset...')
    dataset = load_dataset('multi_nli')
    columns = ['premise', 'hypothesis', 'genre', 'label']

    print('Saving as csv\'s... ')
    for dset in dataset.keys():
            # Weird bug here that adds 'unnamed 0 index kinda row to the final product, plss fix
            pd.DataFrame(list(zip(*[dataset[dset][i] for i in columns])), columns=columns).to_csv(f'data/raw/{dset}.csv')

def dataset_preprocessor(data_path):
     # 1. Cast to tensors
     # 2. Tokenize in some way
     # 3. 
    
    data = pd.read_csv(f'data/raw/{data_path}.csv')
    tokenizer = torchtext.data.get_tokenizer('basic_english')

    data[['premise', 'hypothesis']] = data[['premise', 'hypothesis']].astype(str).map(tokenizer)

    # So needlessly complicated because pandas is stupid and keeps the index
    vocab = torchtext.vocab.build_vocab_from_iterator([i[1] for i in data['premise']._append(data['hypothesis'],
                                                                                              ignore_index=True).astype(str).items()])
    
    data[['premise', 'hypothesis']] = data[['premise', 'hypothesis']].map(vocab)



if __name__ == '__main__':
    # something seomthing, if no raw files, run das_huggingface_pirater
    if not all([os.path.exists(f'./data/raw/{i}.csv') for i in ['train', 'validation_matched', 'validation_mismatched']]):
        print('Not all raw datasets detected, pirating all again')
        das_huggingface_pirater()

    dataset_preprocessor('train')