import torch
import lightning as L
from torch.utils.data import DataLoader 
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import LearningRateMonitor
from models.model import IronManWhenHeIsStruckByThorInThatAvengersMovieNotTheSecondObviouslyTheFirst
from fastapi import FastAPI
import wandb
import click
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule

app = FastAPI()
wandb.login(key="7ea086a098e40728fdf48b616051776a17daf566")

# def train_model():
#     trainer = L.Trainer(
#         max_epochs=5,
#         accelerator="auto",
#         devices=1,
#         logger=[CSVLogger(save_dir="logs/"), WandbLogger(project="MLOps-Project")],
#         callbacks=[LearningRateMonitor(logging_interval="step")],
#     )

#     trainer.fit(model, train_dataloader, val_dataloaders=None)
#     trainer.test(model, test_dataloader)


def cli_main():
    cli = LightningCLI()


if __name__ == "__main__":
    cli_main()
    