from datasets import load_dataset
import numpy as np
import torch


def data_preprocessor():
    data = load_dataset('fashion_mnist')
    #torch.save(data, 'data/raw/fashion_mnist.pt')
    
    # Repeat for both test and train...
    for st in data.keys():
        inputs = torch.tensor(np.array(data[st]['image'])).float()
        labels = torch.tensor(np.array(data[st]['label']))

        inputs = torch.nn.functional.normalize(inputs)
        
        tensor_datset = torch.utils.data.TensorDataset(inputs, labels)

        torch.save(tensor_datset, f'data/processed/{st}.pt')


if __name__ == '__main__':
    data_preprocessor()
