import torch
import torch.nn as nn
import lightning as L
from torchmetrics.functional import accuracy
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

# Lightning code inspired by https://github.com/Lightning-AI/tutorials/blob/main/lightning_examples/cifar10-baseline/baseline.py
# Layer structure graciously stolen from https://www.kaggle.com/code/pankajj/fashion-mnist-with-pytorch-93-accuracy

class FashionMnistModel(torch.nn.Module):
    def __init__(self, in_features=(28,28), out_features=10) -> None:
        """Reuse from above to debug Lightningcli

        Args:
            in_features: Size of input layer very important I swear
            out_features: Should not be called features I know, number of classes in dataset 
            lr: Learning rate of adam optimizer used
        
        Returns:
            torch.tensor
        """

        
        super().__init__()
        
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
            nn.Linear(in_features=120, out_features=out_features)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layers(input.unsqueeze(1))


def create_model(input_size, output_size):
    return FashionMnistModel(in_features=input_size, out_features=output_size)

class IronManWhenHeIsStruckByThorInThatAvengersMovieNotTheSecondObviouslyTheFirst(L.LightningModule):
    def __init__(self, in_features: (int, int) = (28,28), out_features: int = 10, lr: float =0.001):
        """You know? When he gets all powered up and shit?

        Args:
            in_features: Size of input layer very important I swear
            out_features: Should not be called features I know, number of classes in dataset 
            lr: Learning rate of adam optimizer used
        
        Returns:
            torch.tensor
        """

        super().__init__()

        self.save_hyperparameters()
        self.model = FashionMnistModel(in_features, out_features)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = lr


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.to(torch.long))
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y.to(torch.long))
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
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )
        return {"optimizer": optimizer}

    def train_dataloader(self):
        train_data = torch.load('data/processed/train.pt')
        return DataLoader(train_data, batch_size=16)

    def test_dataloader(self):
        test_data =  torch.load('data/processed/test.pt')
        return DataLoader(test_data, batch_size=16)

    def predict_dataloader(self):
        predict_data =  torch.load('data/processed/test.pt')
        return DataLoader(predict_data, batch_size=16)
