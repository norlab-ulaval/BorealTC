import lightning as L
import torch
import torch.nn as nn


class LSTMTerrain(L.LightningModule):
    def __init__(self, lr):
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
