import lightning as L
import torch
import torch.nn as nn
import torchmetrics


class LSTMTerrain(L.LightningModule):
    def __init__(self, lr: float):
        super().__init__()
        self.lr = lr
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        self.linear = nn.Linear(64, 1)

    def forward(self, x):
        ...

    def training_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class CNNTerrain(L.LightningModule):
    def __init__(self, in_size: int, num_filters: int, filter_size: int, num_classes: int, n_wind: int, n_freq: int,
                 lr: float):
        super().__init__()
        self.n_wind = n_wind
        self.n_freq = n_freq

        self.lr = lr
        self.in_layer = nn.Conv2d(in_size, in_size, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(in_size)
        self.conv2d1 = nn.Conv2d(in_size, num_filters, kernel_size=filter_size, padding='same')
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_filters * self.n_wind * self.n_freq, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.loss = nn.CrossEntropyLoss()

        self._train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self._val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.batch_norm(x)
        x = self.conv2d1(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x

    def training_step(self, batch, batch_idx):
        x, target = batch
        y = self(x)
        loss = self.loss(y, target)
        acc = self._train_accuracy(y, target)

        self.log('train_loss', loss)
        self.log('train_accuracy', acc)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, target = val_batch
        y = self(x)
        loss = self.loss(y, target)
        acc = self._val_accuracy(y, target)

        self.log('val_loss', loss)
        self.log('val_accuracy', acc)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
