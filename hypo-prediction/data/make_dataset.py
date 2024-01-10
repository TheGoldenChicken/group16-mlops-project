import pandas as pd
import torch
from datasets import load_dataset
import os
import torchtext
from transformers import AutoTokenizer
import itertools

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

    data = pd.read_csv(f'data/raw/{data_path}.csv')
    tokenizer = torchtext.data.get_tokenizer('basic_english')

    data[['premise', 'hypothesis']] = data[['premise', 'hypothesis']].astype(str).map(tokenizer)

    combined_with_delim = [i + ['[CLS]'] + r +  ['[EOS]'] for i, r in zip(data['premise'], data['hypothesis'])]


    # if use_vocab:
    #     # vocab = torchtext.vocab.build_vocab_from_iterator(combined_with_delim, specials=['[UNK]', '[CLS]', '[EOS]'])
    #     combined_with_delim = list(map(vocab, combined_with_delim))
    #     torch.save(combined_with_delim, f'./data/processed/{data_path}.pt')
    #     torch.save

        # return vocab, combined_with_delim
    

    torch.save(combined_with_delim, f'./data/processed/{data_path}-input.pt')
    torch.save(list(data['label']), f'./data/processed/{data_path}-targets.pt')
    torch.save(list(data['genre']), f'./data/processed/{data_path}-genre.pt')
    

def make_extended_vocab(data, model_name='bert'):
    """
    Assume 'data' is file loadable by pytorch that is in some way a list of lists of tokenized words  
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    data = torch.load(data)
    tokenizer.add_tokens
    new_words = set(itertools.chain(*data))

if __name__ == '__main__':

    # All datasets in the huggingface multi_nli library
    sets = ['train', 'validation_matched', 'validation_mismatched']
    if not all([os.path.exists(f'./data/raw/{i}.csv') for i in sets]):
        print('Not all raw datasets present, pirating all again')
        das_huggingface_pirater()
        
    print('Processing data...')
    for name in sets:
        dataset_preprocessor(name)