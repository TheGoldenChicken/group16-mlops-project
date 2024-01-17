import torch
from fastapi import FastAPI
from torch.utils.data import DataLoader

app = FastAPI()


def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)

@app.get("/predict")
def predict_endpoint(img):
    dataloader = DataLoader([img], batch_size=1)
    model = torch.load('models/model.pt')
    return predict(model, dataloader)