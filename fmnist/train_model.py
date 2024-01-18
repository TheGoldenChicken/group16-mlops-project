import torch
import lightning as L
from torch.utils.data import DataLoader 
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import LearningRateMonitor
from models.model import IronManWhenHeIsStruckByThorInThatAvengersMovieNotTheSecondObviouslyTheFirst
import wandb

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


if __name__ == "__main__":
    train_model()