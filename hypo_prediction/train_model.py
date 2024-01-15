from transformers import Trainer, TrainingArguments, BertTokenizerFast, AutoTokenizer
from torch.utils.data import DataLoader 
from torch import load, tensor
import pandas as pd
import evaluate
import numpy as np
# def make_dataloader(data_path, tokenizer_path, batch_size=16, train=True):
#     data = pd.read_csv({data_path})[['hypothesis', 'premise', 'label']].values

#     tokenizer = load(tokenizer_path) # Should be huggingface tokenizer class!

#     def collate_fn(input):
#         hypo, pred, lab = ([i[r] for i in input] for r in range(3))
#         output = tokenizer(text=hypo, text_pair=pred, padding="max_length", truncation=True, return_tensors='pt')
#         output['labels'] = tensor(lab)
#         return output

#     data_loader = DataLoader(dataset=data, batch_size=batch_size, collate_fn=collate_fn, shuffle=train)

#     return data_loader

# def make_huggingface_dataset(data_path, tokenizer_path):
#     tokenizer = load(tokenizer_path)
#     data = load(data_path)

#     tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

#     def tokenize_function(input):
#         return tokenizer(text=input['premise'], text_pair=input['hypothesis'], padding="max_length", truncation=True)
    
#     train_set = data['train'].map(tokenize_function)
#     eval_set = data['validation_matched']

#     return train_set, eval_set



def train_model(model, data_path, tokenizer_path, train_test=0.9):
    training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
    metric = evaluate.load("accuracy")
    tokenizer = AutoTokenizer.from_pretrained('models/bert-base-uncased-extended')
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    train_set = load('data/processed/tokenized_train.pt')
    eval_set = load('data/processed/tokenized_validation_matched.pt')



    from transformers import BertForSequenceClassification, AdamW

    # model = bert-base-uncased
    model = BertForSequenceClassification.from_pretrained(model, num_labels=3)
    # optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    model.resize_token_embeddings(len(tokenizer))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        compute_metrics=compute_metrics,
    )

    trainer.train()



if __name__ == "__main__":
    train_model('bert-base-uncased', data_path='data/raw/multi_nli_dataset.pt' ,tokenizer_path='models/bert-base-uncased_extended_tokenizer.pt', train_test=0.9)