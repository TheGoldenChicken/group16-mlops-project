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
            # Weird bug here that adds 'unnamed 0 index kinda row to the final product, plss fix
            pd.DataFrame(list(zip(*[dataset[dset][i] for i in columns])), columns=columns).to_csv(f'data/raw/{dset}.csv')

    elif save_format == 'huggingface':
        save(dataset, 'data/raw/multi_nli_dataset.pt')

def dataset_preprocessor(data_path, train_test_split=0.9):
    # from torch.utils.data import DataLoader
    # data = pd.read_csv(f'data/raw/{data_path}.csv')
    # tokenizer = get_tokenizer('basic_english')

    # # data[['premise', 'hypothesis']] = data[['premise', 'hypothesis']].astype(str).map(tokenizer)

    # data = data.loc[:, ['premise', 'hypothesis', 'label']]
    # dat_lod  = DataLoader(data.values, batch_size=16, collate_fn=collate_fn, shuffle=False)
    # next(iter(dat_lod))
    # combined_with_delim = [['[CLS]'] + i + ['[SEP]'] + r + ['[SEP]']  for i, r in zip(data['premise'], data['hypothesis'])]

    # save(combined_with_delim, f'./data/processed/{data_path}-input.pt')
    # save(list(data['label']), f'./data/processed/{data_path}-targets.pt')
    # save(list(data['genre']), f'./data/processed/{data_path}-genre.pt')
    data = pd.read_csv(data_path)
    train_set_size = int(len(data))*0.9 
    train_set = data.loc[:train_set_size]
    test_set = data.loc[train_set_size:]
    train_set.to_csv('data/raw/train_set.csv')
    test_set.to_csv('data/raw/test_set.csv')

    # If anything, this funciton should be something that automatically maps sentences to tokens to save time during training?
    
def make_extended_vocab(data_path, model_name='bert'):
    """
    Assume 'data' is file loadable by pytorch that is in some way a list of lists of tokenized words  
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    data = load(data_path)
    new_words = list(set(itertools.chain(*data)))
    tokenizer.add_tokens(new_words)

    save(tokenizer, f'models/{model_name}_extended_tokenizer.pt')

 
if __name__ == '__main__':
    # All datasets in the huggingface multi_nli library
    sets = ['train', 'validation_matched', 'validation_mismatched']
    das_huggingface_pirater('huggingface')
    # if not all([os.path.exists(f'./data/raw/{i}.csv') for i in sets]):
    #     print('Not all raw datasets present, pirating all again')
    #     das_huggingface_pirater()
        
    # print('Processing data...')
    # for name in sets:
    #     dataset_preprocessor(name)

    model_name = 'bert-base-uncased'
    data_path = './data/processed/train-input.pt'
    print(f'Extending tokenizer for {model_name} with words from {data_path}')
    make_extended_vocab(data_path=data_path, model_name=model_name)