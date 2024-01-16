import torch
import torch.nn as nn
import lightning as L
from torchmetrics.functional import accuracy
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from lightning.pytorch.callbacks import LearningRateMonitor

class FashionMnistModel(torch.nn.Module):
    def __init__(self, in_features=(28,28), out_features=10):
        super().__init__()

        # Layer structure graciously stolen from https://www.kaggle.com/code/pankajj/fashion-mnist-with-pytorch-93-accuracy

        self.layers = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(in_features=64*6*6, out_features=600),
            nn.Dropout1d(0.25),
            nn.Linear(in_features=600, out_features=120),
            nn.Linear(in_features=120, out_features=10)
        )

    def forward(self, input):
        return self.layers(input.unsqueeze(1))

class IronManWhenHeIsStruckByThorInThatAvengersMovieNotTheSecondObviouslyTheFirst(L.LightningModule):
    def __init__(self, in_features=(28,28), out_features=10, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = FashionMnistModel(in_features, out_features)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        if torch.any(torch.isnan(loss)):
            i = 2

        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.005,
            momentum=0.9,
            weight_decay=5e-4,
        )
        return {"optimizer": optimizer}
        

    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(
    #         self.parameters(),
    #         lr=self.hparams.lr,
    #         momentum=0.9,
    #         weight_decay=5e-4,
    #     )
    #     steps_per_epoch = 45000 // BATCH_SIZE
    #     scheduler_dict = {
    #         "scheduler": OneCycleLR(
    #             optimizer,
    #             0.1,
    #             epochs=self.trainer.max_epochs,
    #             steps_per_epoch=steps_per_epoch,
    #         ),
    #         "interval": "step",
    #     }
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


