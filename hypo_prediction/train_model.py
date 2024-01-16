from torch.utils.data import DataLoader 
from torch import load, tensor
import pandas as pd
import numpy as np
import torch
import lightning as L
from torchmetrics.functional import accuracy
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
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