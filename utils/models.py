from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import numpy as np
import pandas as pd
import scipy.stats as scst
import torch
import torch.nn as nn
import torchmetrics
from sklearn.svm import LinearSVC

from utils.constants import ch_cols

if TYPE_CHECKING:
    ExperimentData = dict[str, pd.DataFrame | np.ndarray]


class LSTMTerrain(L.LightningModule):
    def __init__(self, lr: float):
        super().__init__()
        self.lr = lr
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=64, num_layers=1, batch_first=True
        )
        self.linear = nn.Linear(64, 1)

    def forward(self, x):
        ...

    def training_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class CNNTerrain(L.LightningModule):
    def __init__(
        self,
        in_size: int,
        num_filters: int,
        filter_size: int,
        num_classes: int,
        n_wind: int,
        n_freq: int,
        lr: float,
    ):
        super().__init__()
        self.n_wind = n_wind
        self.n_freq = n_freq

        self.lr = lr
        self.in_layer = nn.Conv2d(in_size, in_size, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(in_size)
        self.conv2d1 = nn.Conv2d(
            in_size, num_filters, kernel_size=filter_size, padding="same"
        )
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_filters * self.n_wind * self.n_freq, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.loss = nn.CrossEntropyLoss()

        self._train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self._val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )

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

        self.log("train_loss", loss)
        self.log("train_accuracy", acc)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, target = val_batch
        y = self(x)
        loss = self.loss(y, target)
        acc = self._val_accuracy(y, target)

        self.log("val_loss", loss)
        self.log("val_accuracy", acc)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


def support_vector_machine(
    train_dat: list[ExperimentData],
    test_dat: list[ExperimentData],
    summary: pd.DataFrame,
    n_stat_mom: int,
    svm_train_opt: dict,
):
    kernel_func = svm_train_opt["kernel_function"]
    polynomial_order = svm_train_opt["polynomial_order"]
    kernel_scale = svm_train_opt["kernel_scale"]
    box_constraint = svm_train_opt["box_constraint"]
    standardize = svm_train_opt["standardize"]
    coding = svm_train_opt["coding"]

    # Highest sampling frequency
    hf_sensor = summary["sampling_freq"].idxmax()
    # Other sensors are low frequency
    lf_sensors = tuple(sens for sens in summary.index.values if sens != hf_sensor)

    def stat_moments(data: np.array) -> np.array:
        """Apply statistical moments to data

        Args:
            data (np.array): A numpy array of shape (Nwind, Window length, Channels)

        Returns:
            np.array: numpy array of size (Nwind, Channels, Moments)
        """
        # terrain, terr_idx, run_idx, win_idx, time, <sensor_channels>
        sens_data = data[:, :, 5:].astype(float)
        mean = np.mean(sens_data, axis=1)
        std = np.std(sens_data, axis=1)
        skew = scst.skew(sens_data, axis=1)
        # Use Pearson kurtosis
        kurt = scst.kurtosis(sens_data, axis=1, fisher=False)

        moments = np.stack([mean, std, skew, kurt], axis=2)

        return moments

    def convert_to_moments(data: ExperimentData) -> np.array:
        """Merge sensor data by applying statistical moments to them

        Args:
            data (ExperimentData): _description_

        Returns:
            _type_: _description_
        """
        lf_moments = np.hstack([stat_moments(data[sens]) for sens in lf_sensors])
        moments = np.hstack([stat_moments(data[hf_sensor]), lf_moments])
        stat_moms = moments[:, :, :n_stat_mom]
        n_channels = stat_moms.shape[1]

        X = stat_moms.reshape(-1, n_stat_mom * n_channels, order="F")
        y = data[hf_sensor][:, 0, ch_cols["terr_idx"]]

        for i in range(n_stat_mom):
            idx = i * n_channels
            assert (
                stat_moms[:, :, i] == X[:, idx : idx + n_channels]
            ).all(), "Unconsistent number of channels"

        return X, y

    for K_idx, (K_train, K_test) in enumerate(zip(train_dat, test_dat)):
        K_XTrain, K_YTrain = convert_to_moments(K_train)
        K_XTest, K_YTest = convert_to_moments(K_test)

        svm = LinearSVC()

        # assert()

        print(
            K_XTrain.shape,
            K_XTest.shape,
            K_YTrain.shape,
            K_YTest.shape,
        )

        # n_channels =
        # assert()
