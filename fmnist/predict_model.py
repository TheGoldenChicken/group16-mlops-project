import torch
from fastapi import FastAPI, UploadFile, File
from fmnist.models.model import IronManWhenHeIsStruckByThorInThatAvengersMovieNotTheSecondObviouslyTheFirst
import io

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

@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    # transform = transforms.ToTensor()
    # img = transform(img)
    file_bytes = await file.read()
    img_tensor = torch.load(io.BytesIO(file_bytes))

    # Print shape max and min
    print(img_tensor.shape)
    print(img_tensor.max())
    print(img_tensor.min())
    
    model = IronManWhenHeIsStruckByThorInThatAvengersMovieNotTheSecondObviouslyTheFirst(img_tensor)
    model.load_state_dict(torch.load("models/model.pt"))

    return model(img_tensor).argmax().item()