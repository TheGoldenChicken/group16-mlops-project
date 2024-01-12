from hypo_prediction.data.make_dataset import make_extended_vocab
import os
import torch
from transformers import AutoTokenizer


def test_make_extended_vocab():
    """Checks if the tokenizer is saved correctly,
    and if the vocab has been exstended."""

    model_name = 'bert-base-uncased'
    make_extended_vocab('./data/processed/train-input.pt', model_name)

    assert os.path.exists('./models/bert-base-uncased_extended_tokenizer.pt'), 'The tokenizer has not been saved'

    # Assert that the tokenizer can be loaded
    assert torch.load('./models/bert-base-uncased_extended_tokenizer.pt'), 'The tokenizer cannot be loaded with torch.load()'

    # Assert that the vocab has been extended
    base_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = torch.load('./models/bert-base-uncased_extended_tokenizer.pt')
    assert len(tokenizer) > len(base_tokenizer), 'The vocab has not been extended'