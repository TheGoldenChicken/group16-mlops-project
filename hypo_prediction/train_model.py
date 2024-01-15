from transformers import Trainer, TrainingArguments 
from torch.utils.data import DataLoader 
from torch import load, tensor
import pandas as pd

def make_dataloader(data_path, tokenizer_path, batch_size=16, train=True):
    data = pd.read_csv({data_path})[['hypothesis', 'premise', 'label']].values

    tokenizer = load(tokenizer_path) # Should be huggingface tokenizer class!

    def collate_fn(input):
        hypo, pred, lab = ([i[r] for i in input] for r in range(3))
        output = tokenizer(text=hypo, text_pair=pred, padding="max_length", truncation=True, return_tensors='pt')
        output['labels'] = tensor(lab)
        return output

    data_loader = DataLoader(dataset=data, batch_size=batch_size, collate_fn=collate_fn, shuffle=train)

    return data_loader

def make_huggingface_dataset(data_path, tokenizer_path):
    tokenizer = load(tokenizer_path)
    data = load(data_path).values

    def tokenize_function(input):
        return tokenizer(text=input['premise'], text_pair=input['hypothesis'], padding="max_length", truncation=True)
    
    train_set = data['train'].map(tokenize_function)
    eval_set = data['validation_matched']

    return train_set, eval_set



def train_model(model, train_data, val_data, tokenizer):

    train_path = 'data/raw/train.csv'
    val_path = 'data/raw/validation_matched.csv'
    tokenizer_path = 'models/bert-base-uncased_extended_tokenizer.pt'

    train_loader = make_dataloader(train_path, tokenizer_path)
    train_loader = make_dataloader(val_path, tokenizer_path, train=False)
    


if __name__ == "__main__":
    make_dataloader('data/processed/train', 'models/bert-base-uncased_extended_tokenizer.pt', train=True)
    make_dataloader('data/processed/train', 'models/bert-base-uncased_extended_tokenizer.pt', train=False)
