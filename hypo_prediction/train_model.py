from torch.utils.data import DataLoader 
from torch import load


def make_dataloader(data_path, tokenizer_path):
    dat_input = load(f'{data_path}-input.pt')
    dat_targets = load(f'{data_path}-targets.pt')
    tokenizer = load(tokenizer_path)




    data_loader = DataLoader()

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# def train_model(model, data, tokenizer):


if __name__ == "__main__":
    make_dataloader('data/processed/train', 'models/bert-base-uncased_extended_tokenizer.pt')
