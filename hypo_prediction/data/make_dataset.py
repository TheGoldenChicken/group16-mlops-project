import os
import itertools
import pandas as pd
from torch import load, save
from datasets import load_dataset
from torchtext.data import get_tokenizer
from transformers import AutoTokenizer


def das_huggingface_pirater(save_format):
    """Pirates the huggingface library
    Yeah no it doesn't really pirate it, but you know"""

    print('Loading multi_nli dataset...')
    dataset = load_dataset('multi_nli')
    columns = ['premise', 'hypothesis', 'genre', 'label']

    if save_format == 'pd':
        print('Saving as csv\'s... ')
        for dset in dataset.keys():
            pd.DataFrame(list(zip(*[dataset[dset][i] for i in columns])), columns=columns).to_csv(f'data/raw/{dset}.csv')

    elif save_format == 'huggingface':
        save(dataset, 'data/raw/multi_nli_dataset.pt')

def dataset_preprocessor(data_path, tokenizer_path, train_test_split=0.9):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # tokenizer = load(tokenizer_path)
    data = load(data_path)

    def tokenize_function(input):
        return tokenizer(text=input['premise'], text_pair=input['hypothesis'], padding="max_length", truncation=True)

#     # save(combined_with_delim, f'./data/processed/{data_path}-input.pt')
#     # save(list(data['label']), f'./data/processed/{data_path}-targets.pt')
#     # save(list(data['genre']), f'./data/processed/{data_path}-genre.pt')
#     data = pd.read_csv(data_path)
#     train_set_size = int(len(data))*0.9 
#     train_set = data.loc[:train_set_size]
#     test_set = data.loc[train_set_size:]
#     train_set.to_csv('data/raw/train_set.csv')
#     test_set.to_csv('data/raw/test_set.csv')
    for st in data.keys():
        save(data[st].map(tokenize_function), f'data/processed/tokenized_{st}.pt') 

    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        
def make_extended_vocab(data_path, model_name='bert'):
    """
    Assume 'data' is file loadable by pytorch that is in some way a list of lists of tokenized words  
    """
    data = load(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = tokenizer.train_new_from_iterator(data['train']['premise'] + data['train']['hypothesis'], vocab_size=90000)
    tokenizer.save_pretrained(f'models/{model_name}-extended-tokenizer')

    # save(tokenizer, f'models/{model_name}_extended_tokenizer.pt')
    # pre_tokenizer = get_tokenizer('basic_english')
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # data = load(data_path)
    # new_words = set(itertools.chain(*list(map(pre_tokenizer, data['train']['hypothesis']))))
    # new_words = new_words.union(set(itertools.chain(*list(map(pre_tokenizer, data['train']['premise'])))))
    # new_words = list(new_words)
    # tokenizer.add_tokens(new_words)
    # tokenizer.save(f'models/{model_name}_extended_tokenizer.json')
    # save(tokenizer, f'models/{model_name}_extended_tokenizer.pt')

 
if __name__ == '__main__':
    # All datasets in the huggingface multi_nli library
    # das_huggingface_pirater('huggingface')
        
    model_name = 'bert-base-uncased'
    data_path = './data/raw/multi_nli_dataset.pt'
    print(f'Extending tokenizer for {model_name} with words from {data_path}')
    make_extended_vocab(data_path=data_path, model_name=model_name)
    
    print('Processing data...')
    dataset_preprocessor('./data/raw/multi_nli_dataset.pt', 'models/bert-base-uncased-extended-tokenizer')
