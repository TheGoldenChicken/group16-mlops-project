import torch
import lightning as L
from torch.utils.data import DataLoader 
<<<<<<< HEAD:fashion_mnist/train_model.py
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
=======
from torchmetrics.functional import accuracy
from lightning.pytorch.loggers import CSVLogger
>>>>>>> clean model code:hypo_prediction/train_model.py
from lightning.pytorch.callbacks import LearningRateMonitor
from models.model import IronManWhenHeIsStruckByThorInThatAvengersMovieNotTheSecondObviouslyTheFirst
from fastapi import FastAPI
import wandb

app = FastAPI()
wandb.login(key="7ea086a098e40728fdf48b616051776a17daf566")


def train_model():
    train_data = torch.load('data/processed/train.pt')
    test_data = torch.load('data/processed/test.pt')

    train_dataloader = DataLoader(train_data, batch_size=16)
    test_dataloader = DataLoader(test_data, batch_size=16)

    model = IronManWhenHeIsStruckByThorInThatAvengersMovieNotTheSecondObviouslyTheFirst(lr=0.005)

    trainer = L.Trainer(
        max_epochs=5,
        accelerator="auto",
        devices=1,
        logger=[CSVLogger(save_dir="logs/"), WandbLogger(project="MLOps-Project")],
        callbacks=[LearningRateMonitor(logging_interval="step")],
    )

    trainer.fit(model, train_dataloader, val_dataloaders=None)
    trainer.test(model, test_dataloader)

@app.get("/train")
def train_model_endpoint():
    train_model()


if __name__ == "__main__":
    train_model()