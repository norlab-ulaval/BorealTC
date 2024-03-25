from __future__ import annotations

import time
from typing import TYPE_CHECKING

import einops as ein
import lightning as L
import numpy as np
import pandas as pd
import pipeline as pp
import scipy.stats as scst
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision as tv
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger

from pathlib import Path

from sklearn.multiclass import OutputCodeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils.augmentations import (
    NormalizeMCS,
    SpectralCutout,
    SpectralNoise,
)
from utils.constants import ch_cols, imu_dim, pro_dim
from utils.datamodule import (
    MCSDataModule,
    TemporalDataModule,
    MambaDataModule,
    MambaDataModuleCombined,
)

from mamba_ssm.models.mixer_seq_simple import create_block

if TYPE_CHECKING:
    ExperimentData = dict[str, pd.DataFrame | np.ndarray]


class LSTMTerrain(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        convolutional: bool,
        num_classes: int,
        lr: float,
        conv_num_filters: int = 5,
        learning_rate_factor: float = 0.1,
        reduce_lr_patience: int = 8,
        class_weights: list[float] | None = None,
        focal_loss: bool = False,
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.convolutional = convolutional
        self.conv_num_filters = conv_num_filters
        self.learning_rate_factor = learning_rate_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.class_weights = class_weights
        self.focal_loss = focal_loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.lr = lr

        self.conv = None
        self.bn = None
        self.fc_conv = None
        if convolutional:
            self.conv = nn.Conv1d(
                in_channels=input_size,
                out_channels=conv_num_filters,
                kernel_size=10,
                padding="same",
            )
            self.bn = nn.BatchNorm1d(conv_num_filters)
            self.fc_conv = nn.Linear(conv_num_filters, hidden_size)
        self.rnn = nn.LSTM(
            input_size if not convolutional else hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(
            hidden_size * (2 if bidirectional else 1) * num_layers, num_classes
        )
        self.softmax = nn.Softmax(dim=1)

        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights) if class_weights else None
        )

        self._train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self._val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self._test_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self._val_classifications = [
            {"pred": [], "true": [], "ftime": [], "ptime": [], "conf": []}
        ]
        self._test_classifications = [
            {"pred": [], "true": [], "ftime": [], "ptime": [], "conf": []}
        ]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,
        #                                                       lr_lambda=lambda epoch: self.learning_rate_factor,
        #                                                       verbose=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.reduce_lr_patience,
            factor=self.learning_rate_factor,
            verbose=True,
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                monitor="val_loss",
            ),
        )

    @property
    def val_classification(self):
        return (
            self._val_classifications[-2] if len(self._val_classifications) > 1 else {}
        )

    @property
    def test_classification(self):
        return (
            self._test_classifications[-2]
            if len(self._test_classifications) > 1
            else {}
        )

    def forward(self, x):
        if self.convolutional:
            x = ein.rearrange(x, "b c n -> b n c")
            x = self.conv(x)
            x = self.bn(x)
            x = ein.rearrange(x, "b n c -> b c n")
            x = self.fc_conv(x)

        output, (hn, cn) = self.rnn(x)
        x = ein.rearrange(hn, "l n c -> n (l c)")
        x = self.fc(x)
        return x

    def loss(self, y, target):
        if self.focal_loss:
            return tv.ops.sigmoid_focal_loss(
                y,
                F.one_hot(target, self.num_classes).to(torch.float),
                alpha=self.focal_loss_alpha,
                reduction="mean",
                gamma=self.focal_loss_gamma,
            )
        return self.ce_loss(y, target)

    def training_step(self, batch, batch_idx):
        x, target = batch
        pred = self(x)
        loss = self.loss(pred, target)
        acc = self._train_accuracy(pred, target)

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_accuracy", acc, on_step=True)

        return loss

    def on_train_epoch_end(self):
        self.log(
            "train_accuracy_epoch",
            self._train_accuracy.compute(),
            prog_bar=True,
            on_epoch=True,
        )
        self._train_accuracy.reset()

    def validation_step(self, val_batch, batch_idx):
        x, target = val_batch
        pred = self(x)
        y = self.softmax(pred)
        loss = self.loss(pred, target)
        acc = self._val_accuracy(pred, target)

        self.log("val_loss", loss, on_step=True)
        self.log("val_accuracy", acc, on_step=True)

        pred_classes = torch.argmax(y, dim=1)
        self._val_classifications[-1]["pred"].append(
            pred_classes.detach().cpu().numpy()
        )
        self._val_classifications[-1]["true"].append(target.detach().cpu().numpy())
        self._val_classifications[-1]["conf"].append(y.detach().cpu().numpy())

        return loss

    def on_validation_epoch_end(self):
        self.log(
            "val_accuracy_epoch",
            self._val_accuracy.compute(),
            prog_bar=True,
            on_epoch=True,
        )
        self._val_accuracy.reset()

        self._val_classifications[-1]["pred"] = np.hstack(
            self._val_classifications[-1]["pred"]
        )
        self._val_classifications[-1]["true"] = np.hstack(
            self._val_classifications[-1]["true"]
        )
        self._val_classifications[-1]["conf"] = np.vstack(
            self._val_classifications[-1]["conf"]
        )
        self._val_classifications.append(
            {"pred": [], "true": [], "ftime": [], "ptime": [], "conf": []}
        )

    def test_step(self, test_batch, batch_idx):
        x, target = test_batch
        pred = self(x)
        y = self.softmax(pred)
        loss = self.loss(pred, target)
        acc = self._test_accuracy(pred, target)

        self.log("test_loss", loss, on_step=True)
        self.log("test_accuracy", acc, on_step=True)

        pred_classes = torch.argmax(y, dim=1)
        self._test_classifications[-1]["pred"].append(
            pred_classes.detach().cpu().numpy()
        )
        self._test_classifications[-1]["true"].append(target.detach().cpu().numpy())
        self._test_classifications[-1]["conf"].append(y.detach().cpu().numpy())

        return loss

    def on_test_epoch_end(self):
        self.log(
            "test_accuracy_epoch",
            self._test_accuracy.compute(),
            prog_bar=True,
            on_epoch=True,
        )
        self._test_accuracy.reset()

        self._test_classifications[-1]["pred"] = np.hstack(
            self._test_classifications[-1]["pred"]
        )
        self._test_classifications[-1]["true"] = np.hstack(
            self._test_classifications[-1]["true"]
        )
        self._test_classifications[-1]["conf"] = np.vstack(
            self._test_classifications[-1]["conf"]
        )
        self._test_classifications.append(
            {"pred": [], "true": [], "ftime": [], "ptime": [], "conf": []}
        )


class CNNTerrain(L.LightningModule):
    def __init__(
        self,
        in_size: int,
        num_filters: int,
        filter_size: int | tuple[int, int],
        num_classes: int,
        n_wind: int,
        n_freq: int,
        lr: float,
        learning_rate_factor: float = 0.1,
        reduce_lr_patience: int = 8,
        class_weights: list[float] | None = None,
        focal_loss: bool = False,
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2,
        scheduler: str = "plateau",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_wind = n_wind
        self.n_freq = n_freq
        self.num_classes = num_classes
        self.learning_rate_factor = learning_rate_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.class_weights = class_weights
        self.focal_loss = focal_loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.lr = lr
        self.scheduler = scheduler
        self.dropout = dropout

        self.in_layer = nn.Conv2d(in_size, in_size, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_size)
        self.drop1 = nn.Dropout2d(p=dropout)
        self.conv2d1 = nn.Conv2d(
            in_size, num_filters, kernel_size=filter_size, padding="same"
        )
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.drop2 = nn.Dropout2d(p=dropout)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_filters * self.n_wind * self.n_freq, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights) if class_weights else None
        )

        self._train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self._val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self._test_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self._val_classifications = [
            {"pred": [], "true": [], "ftime": [], "ptime": [], "conf": []}
        ]
        self._test_classifications = [
            {"pred": [], "true": [], "ftime": [], "ptime": [], "conf": [], "repr": []}
        ]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if self.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                patience=self.reduce_lr_patience,
                factor=self.learning_rate_factor,
                verbose=True,
            )
        elif self.scheduler == "linear":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=self.learning_rate_factor
            )
        # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,
        #                                                       lr_lambda=lambda epoch: self.learning_rate_factor,
        #                                                       verbose=True)
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                monitor="val_loss",
            ),
        )

    @property
    def val_classification(self):
        return (
            self._val_classifications[-2] if len(self._val_classifications) > 1 else {}
        )

    @property
    def test_classification(self):
        return (
            self._test_classifications[-2]
            if len(self._test_classifications) > 1
            else {}
        )

    def forward(self, x):
        x = self.in_layer(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.conv2d1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.drop2(x)

        embed = self.flatten(x)
        x = self.fc(embed)

        return x, embed

    def loss(self, y, target):
        if self.focal_loss:
            return tv.ops.sigmoid_focal_loss(
                y,
                F.one_hot(target, self.num_classes).to(torch.float),
                alpha=self.focal_loss_alpha,
                reduction="mean",
            )
        return self.ce_loss(y, target)

    def training_step(self, batch, batch_idx):
        x, target = batch
        pred, embed = self(x)
        loss = self.loss(pred, target)
        acc = self._train_accuracy(pred, target)

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_accuracy", acc, on_step=True)

        return loss

    def on_train_epoch_end(self):
        self.log(
            "train_accuracy_epoch",
            self._train_accuracy.compute(),
            prog_bar=True,
            on_epoch=True,
        )
        self._train_accuracy.reset()

    def validation_step(self, val_batch, batch_idx):
        x, target = val_batch
        pred, embed = self(x)
        y = self.softmax(pred)
        loss = self.loss(pred, target)
        acc = self._val_accuracy(pred, target)

        self.log("val_loss", loss, on_step=True)
        self.log("val_accuracy", acc, on_step=True)

        pred_classes = torch.argmax(y, dim=1)
        self._val_classifications[-1]["pred"].append(
            pred_classes.detach().cpu().numpy()
        )
        self._val_classifications[-1]["true"].append(target.detach().cpu().numpy())
        self._val_classifications[-1]["conf"].append(y.detach().cpu().numpy())

        return loss

    def on_validation_epoch_end(self):
        self.log(
            "val_accuracy_epoch",
            self._val_accuracy.compute(),
            prog_bar=True,
            on_epoch=True,
        )
        self._val_accuracy.reset()

        self._val_classifications[-1]["pred"] = np.hstack(
            self._val_classifications[-1]["pred"]
        )
        self._val_classifications[-1]["true"] = np.hstack(
            self._val_classifications[-1]["true"]
        )
        self._val_classifications[-1]["conf"] = np.vstack(
            self._val_classifications[-1]["conf"]
        )
        self._val_classifications.append(
            {"pred": [], "true": [], "ftime": [], "ptime": [], "conf": []}
        )

    def test_step(self, test_batch, batch_idx):
        x, target = test_batch
        pred, embed = self(x)
        y = self.softmax(pred)
        loss = self.loss(pred, target)
        acc = self._test_accuracy(pred, target)

        self.log("test_loss", loss, on_step=True)
        self.log("test_accuracy", acc, on_step=True)

        pred_classes = torch.argmax(y, dim=1)
        self._test_classifications[-1]["pred"].append(
            pred_classes.detach().cpu().numpy()
        )
        self._test_classifications[-1]["true"].append(target.detach().cpu().numpy())
        self._test_classifications[-1]["conf"].append(y.detach().cpu().numpy())
        self._test_classifications[-1]["repr"].append(embed.detach().cpu().numpy())

        return loss

    def on_test_epoch_end(self):
        self.log(
            "test_accuracy_epoch",
            self._test_accuracy.compute(),
            prog_bar=True,
            on_epoch=True,
        )
        self._test_accuracy.reset()

        self._test_classifications[-1]["pred"] = np.hstack(
            self._test_classifications[-1]["pred"]
        )
        self._test_classifications[-1]["true"] = np.hstack(
            self._test_classifications[-1]["true"]
        )
        self._test_classifications[-1]["conf"] = np.vstack(
            self._test_classifications[-1]["conf"]
        )
        self._test_classifications[-1]["repr"] = np.vstack(
            self._test_classifications[-1]["repr"]
        )
        self._test_classifications.append(
            {"pred": [], "true": [], "ftime": [], "ptime": [], "conf": [], "repr": []}
        )


class MambaTerrain(L.LightningModule):
    def __init__(
        self,
        d_model_imu: int,
        d_model_pro: int,
        norm_epsilon: float,
        ssm_cfg_imu: dict,
        ssm_cfg_pro: dict,
        out_method: str,
        num_classes: int,
        lr: float,
        learning_rate_factor: float = 0.1,
        reduce_lr_patience: int = 8,
        class_weights: list[float] | None = None,
        focal_loss: bool = False,
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.out_method = out_method
        self.num_classes = num_classes
        self.lr = lr
        self.learning_rate_factor = learning_rate_factor
        self.reduce_lr_patience = reduce_lr_patience
        self.class_weights = class_weights
        self.focal_loss = focal_loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

        self.d_model_imu = d_model_imu
        self.d_model_pro = d_model_pro

        self.imu_in_layer = nn.Linear(imu_dim, d_model_imu)
        self.pro_in_layer = nn.Linear(pro_dim, d_model_pro)

        self.mamba_block_imu = create_block(
            d_model=d_model_imu, ssm_cfg=ssm_cfg_imu, norm_epsilon=norm_epsilon
        )

        self.mamba_block_pro = create_block(
            d_model=d_model_pro, ssm_cfg=ssm_cfg_pro, norm_epsilon=norm_epsilon
        )

        self.out_layer = nn.Linear(d_model_imu + d_model_pro, num_classes)

        self.softmax = nn.Softmax(dim=1)

        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights) if class_weights else None
        )

        self._train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self._val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self._test_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self._val_classifications = [
            {"pred": [], "true": [], "ftime": [], "ptime": [], "conf": []}
        ]
        self._test_classifications = [
            {"pred": [], "true": [], "ftime": [], "ptime": [], "conf": []}
        ]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.reduce_lr_patience,
            factor=self.learning_rate_factor,
            verbose=True,
            mode="max",
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                monitor="val_accuracy_epoch",
            ),
        )

    @property
    def val_classification(self):
        return (
            self._val_classifications[-2] if len(self._val_classifications) > 1 else {}
        )

    @property
    def test_classification(self):
        return (
            self._test_classifications[-2]
            if len(self._test_classifications) > 1
            else {}
        )

    def load_from_checkpoint_transfer_learning(checkpoint_path, num_classes, **kwargs):
        self = MambaTerrain.load_from_checkpoint(checkpoint_path, **kwargs)

        # for param in self.parameters():
        #     param.requires_grad = False

        self.out_layer = nn.Linear(self.d_model_imu + self.d_model_pro, num_classes)
        self.num_classes = num_classes

        self._train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self._val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self._test_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        return self

    def _forward_imu(self, x_imu):
        x_imu = self.imu_in_layer(x_imu)
        x_imu, _ = self.mamba_block_imu(x_imu, None)
        return x_imu

    def _forward_pro(self, x_pro):
        x_pro = self.pro_in_layer(x_pro)
        x_pro, _ = self.mamba_block_pro(x_pro, None)
        return x_pro

    def _out_layer_max_pool(self, x_imu, x_pro):
        x_imu = x_imu.transpose(1, 2)
        x_pro = x_pro.transpose(1, 2)

        x_imu = F.max_pool1d(x_imu, kernel_size=x_imu.shape[2])
        x_pro = F.max_pool1d(x_pro, kernel_size=x_pro.shape[2])

        x_imu = x_imu.squeeze(dim=2)
        x_pro = x_pro.squeeze(dim=2)

        x = torch.cat([x_imu, x_pro], dim=1)
        x = self.out_layer(x)

        return x

    def _out_layer_last_state(self, x_imu, x_pro):
        x_imu = x_imu[:, -1]
        x_pro = x_pro[:, -1]
        x = torch.cat([x_imu, x_pro], dim=1)
        x = self.out_layer(x)
        return x

    def forward(self, x):
        x_imu = self._forward_imu(x["imu"])
        x_pro = self._forward_pro(x["pro"])

        if self.out_method == "max_pool":
            x = self._out_layer_max_pool(x_imu, x_pro)
        elif self.out_method == "last_state":
            x = self._out_layer_last_state(x_imu, x_pro)

        return x

    def loss(self, y, target):
        if self.focal_loss:
            return tv.ops.sigmoid_focal_loss(
                y,
                F.one_hot(target, self.num_classes).to(torch.float),
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="mean",
            )
        return self.ce_loss(y, target)

    def _combined_batch_step(self, batch):
        pred = torch.cat([self(_batch[0]) for _batch in batch])
        target = torch.cat([_batch[1] for _batch in batch])

        return pred, target

    def training_step(self, batch, batch_idx):
        if isinstance(batch[0], list):
            pred, target = self._combined_batch_step(batch)
        else:
            x, target = batch
            pred = self(x)

        loss = self.loss(pred, target)
        acc = self._train_accuracy(pred, target)

        self.log("train_loss", loss, prog_bar=True, on_step=True, batch_size=len(pred))
        self.log("train_accuracy", acc, on_step=True, batch_size=len(pred))

        return loss

    def on_train_epoch_end(self):
        self.log(
            "train_accuracy_epoch",
            self._train_accuracy.compute(),
            prog_bar=True,
            on_epoch=True,
        )
        self._train_accuracy.reset()

    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        x, target = val_batch
        pred = self(x)

        y = self.softmax(pred)
        loss = self.loss(pred, target)
        acc = self._val_accuracy(pred, target)

        self.log("val_loss", loss, on_step=True, batch_size=len(pred))
        self.log("val_accuracy", acc, on_step=True, batch_size=len(pred))

        pred_classes = torch.argmax(y, dim=1)
        self._val_classifications[-1]["pred"].append(
            pred_classes.detach().cpu().numpy()
        )
        self._val_classifications[-1]["true"].append(target.detach().cpu().numpy())
        self._val_classifications[-1]["conf"].append(y.detach().cpu().numpy())

        return loss

    def on_validation_epoch_end(self):
        self.log(
            "val_accuracy_epoch",
            self._val_accuracy.compute(),
            prog_bar=True,
            on_epoch=True,
        )
        self._val_accuracy.reset()

        self._val_classifications[-1]["pred"] = np.hstack(
            self._val_classifications[-1]["pred"]
        )
        self._val_classifications[-1]["true"] = np.hstack(
            self._val_classifications[-1]["true"]
        )
        self._val_classifications[-1]["conf"] = np.vstack(
            self._val_classifications[-1]["conf"]
        )
        self._val_classifications.append(
            {"pred": [], "true": [], "ftime": [], "ptime": [], "conf": []}
        )

    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        x, target = test_batch
        pred = self(x)

        y = self.softmax(pred)
        loss = self.loss(pred, target)
        acc = self._test_accuracy(pred, target)

        self.log("test_loss", loss, on_step=True, batch_size=len(pred))
        self.log("test_accuracy", acc, on_step=True, batch_size=len(pred))

        pred_classes = torch.argmax(y, dim=1)
        self._test_classifications[-1]["pred"].append(
            pred_classes.detach().cpu().numpy()
        )
        self._test_classifications[-1]["true"].append(target.detach().cpu().numpy())
        self._test_classifications[-1]["conf"].append(y.detach().cpu().numpy())

        return loss

    def on_test_epoch_end(self):
        self.log(
            "test_accuracy_epoch",
            self._test_accuracy.compute(),
            prog_bar=True,
            on_epoch=True,
        )
        self._test_accuracy.reset()

        self._test_classifications[-1]["pred"] = np.hstack(
            self._test_classifications[-1]["pred"]
        )
        self._test_classifications[-1]["true"] = np.hstack(
            self._test_classifications[-1]["true"]
        )
        self._test_classifications[-1]["conf"] = np.vstack(
            self._test_classifications[-1]["conf"]
        )
        self._test_classifications.append(
            {"pred": [], "true": [], "ftime": [], "ptime": [], "conf": []}
        )


def mamba_network(
    train_data: list[ExperimentData],
    test_data: list[ExperimentData],
    mamba_par: dict,
    mamba_train_opt: dict,
    ssm_cfg_imu: dict,
    ssm_cfg_pro: dict,
    description: dict,
    custom_callbacks=None,
    random_state: int | None = None,
    test: bool = True,
    checkpoint: Path | None = None,
    logging: bool = True,
) -> dict:
    # Seed
    L.seed_everything(random_state)

    # Mamba parameters
    if custom_callbacks is None:
        custom_callbacks = []
    d_model_imu = mamba_par["d_model_imu"]
    d_model_pro = mamba_par["d_model_pro"]
    norm_epsilon = mamba_par["norm_epsilon"]
    out_method = mamba_train_opt["out_method"]
    num_classes = mamba_train_opt["num_classes"]
    dataset = description["dataset"]

    # Training parameters
    valid_perc = mamba_train_opt["valid_perc"]
    init_learn_rate = mamba_train_opt["init_learn_rate"]
    learn_drop_factor = mamba_train_opt["learn_drop_factor"]
    max_epochs = mamba_train_opt["max_epochs"]
    minibatch_size = mamba_train_opt["minibatch_size"]
    valid_patience = mamba_train_opt["valid_patience"]
    reduce_lr_patience = mamba_train_opt["reduce_lr_patience"]
    valid_frequency = mamba_train_opt["valid_frequency"]
    gradient_treshold = mamba_train_opt["gradient_treshold"]
    focal_loss = mamba_train_opt["focal_loss"]
    focal_loss_alpha = mamba_train_opt["focal_loss_alpha"]
    focal_loss_gamma = mamba_train_opt["focal_loss_gamma"]

    num_workers = 8
    persistent_workers = True

    if "vulpi" in train_data and "husky" in train_data:
        datamodule = MambaDataModuleCombined(
            train_temporal_vulpi=train_data["vulpi"],
            test_temporal_vulpi=test_data["vulpi"],
            train_temporal_husky=train_data["husky"],
            test_temporal_husky=test_data["husky"],
            train_transform=None,
            test_transform=None,
            valid_percent=valid_perc,
            batch_size=minibatch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )
    else:
        datamodule = MambaDataModule(
            train_data,
            test_data,
            train_transform=None,
            test_transform=None,
            train_data_augmentation=None,
            valid_percent=valid_perc,
            batch_size=minibatch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    if checkpoint is not None:
        model = MambaTerrain.load_from_checkpoint_transfer_learning(
            checkpoint_path=checkpoint,
            num_classes=num_classes,
            norm_epsilon=norm_epsilon,
            lr=init_learn_rate,
            learning_rate_factor=learn_drop_factor,
            reduce_lr_patience=reduce_lr_patience,
            class_weights=None,
            focal_loss=focal_loss,
            focal_loss_alpha=focal_loss_alpha,
            focal_loss_gamma=focal_loss_gamma,
        )
    else:
        model = MambaTerrain(
            d_model_imu=d_model_imu,
            d_model_pro=d_model_pro,
            norm_epsilon=norm_epsilon,
            ssm_cfg_imu=ssm_cfg_imu,
            ssm_cfg_pro=ssm_cfg_pro,
            out_method=out_method,
            num_classes=num_classes,
            lr=init_learn_rate,
            learning_rate_factor=learn_drop_factor,
            reduce_lr_patience=reduce_lr_patience,
            class_weights=None,
            focal_loss=focal_loss,
            focal_loss_alpha=focal_loss_alpha,
            focal_loss_gamma=focal_loss_gamma,
        )

    exp_name = f'terrain_classification_mamba_mw_{description["mw"]}_fold_{description["fold"]}_dataset_{dataset}'

    logger = TensorBoardLogger("tb_logs", name=exp_name) if logging else False
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy_epoch", patience=valid_patience, mode="max"
        ),
        *custom_callbacks,
    ]
    if logging:
        checkpoint_folder_path = Path("checkpoints")

        callbacks += [
            DeviceStatsMonitor(),
            LearningRateMonitor(),
            ModelCheckpoint(
                monitor="val_accuracy_epoch",
                dirpath=str(checkpoint_folder_path),
                filename=f"{exp_name}-" + "{epoch:02d}-{val_loss:.6f}",
                save_top_k=1,
                save_last=True,
                mode="max",
            ),
        ]

    trainer = L.Trainer(
        accelerator="gpu",
        precision=32,
        logger=logger,
        log_every_n_steps=1,
        min_epochs=0,
        max_epochs=max_epochs,
        gradient_clip_val=gradient_treshold,
        val_check_interval=valid_frequency,
        callbacks=callbacks,
        enable_checkpointing=logging,
    )

    trainer.fit(model, datamodule)
    loss = trainer.validate(model, datamodule)

    if test:
        trainer.test(model, datamodule)
        out = model.test_classification
    else:
        out = model.val_classification

    out["loss"] = loss
    out["num_params"] = sum(param.numel() for param in model.parameters())

    return out


def convolutional_neural_network(
    train_data: list[ExperimentData],
    test_data: list[ExperimentData],
    cnn_par: dict,
    cnn_train_opt: dict,
    description: dict,
    custom_callbacks=None,
    random_state: int | None = None,
    test: bool = True,
) -> dict:
    # Seed
    L.seed_everything(random_state)

    # CNN parameters
    if custom_callbacks is None:
        custom_callbacks = []
    filter_size = cnn_par["filter_size"]
    num_filters = cnn_par["num_filters"]
    num_classes = cnn_par["num_classes"]
    dataset = description["dataset"]
    mins = description["mins"]
    maxs = description["maxs"]

    # Training parameters
    valid_perc = cnn_train_opt["valid_perc"]
    init_learn_rate = cnn_train_opt["init_learn_rate"]
    learn_drop_factor = cnn_train_opt["learn_drop_factor"]
    max_epochs = cnn_train_opt["max_epochs"]
    minibatch_size = cnn_train_opt["minibatch_size"]
    valid_patience = cnn_train_opt["valid_patience"]
    reduce_lr_patience = cnn_train_opt["reduce_lr_patience"]
    valid_frequency = cnn_train_opt["valid_frequency"]
    gradient_threshold = cnn_train_opt["gradient_threshold"]
    focal_loss = cnn_train_opt.get("focal_loss", False)
    focal_loss_alpha = cnn_train_opt.get("focal_loss_alpha", 0.25)
    focal_loss_gamma = cnn_train_opt.get("focal_loss_gamma", 2)
    dropout = cnn_train_opt.get("dropout", 0.0)
    _, n_freq, n_wind, in_size = train_data["data"].shape
    scheduler = cnn_train_opt.get("scheduler", "plateau")

    checkpoint_path = cnn_train_opt.get("checkpoint_path", None)
    overwrite_final_layer_dim = cnn_train_opt.get("overwrite_final_layer_dim", None)
    use_augmentation = cnn_train_opt.get("use_augmentation", False)
    num_workers = 0
    persistent_workers = False
    verbose = cnn_train_opt.get("verbose", True)

    def to_f32(x):
        return x.astype(np.float32)

    def transpose(x):
        return np.transpose(x, (2, 0, 1))

    train_transform = pp.Bifunctor(
        pp.Compose([transpose, NormalizeMCS(mins, maxs), to_f32]),
        pp.Identity(),
    )
    test_transform = pp.Bifunctor(
        pp.Compose([transpose, NormalizeMCS(mins, maxs), to_f32]),
        pp.Identity(),
    )
    if use_augmentation:
        train_data_augmentation = pp.Bifunctor(
            pp.Compose(
                [
                    SpectralNoise(p_apply=0.3, noise_level=0.05),
                    SpectralCutout(p_apply=0.25, num_mask=5, max_size=4),
                    # SpectralAxialCutout(p_apply=0.25, dim_to_cut=SpectralAxialCutout.CutoutType.CHANNEL, max_num_cut=5),
                    # SpectralAxialCutout(p_apply=0.25, dim_to_cut=SpectralAxialCutout.CutoutType.FREQUENCY, max_num_cut=5),
                    # SpectralAxialCutout(p_apply=0.25, dim_to_cut=SpectralAxialCutout.CutoutType.TIME, max_num_cut=5),
                    to_f32,
                ]
            ),
            pp.Identity(),
        )
    else:
        train_data_augmentation = pp.Identity()

    datamodule = MCSDataModule(
        train_data,
        test_data,
        train_transform,
        test_transform,
        train_data_augmentation,
        valid_percent=valid_perc,
        batch_size=minibatch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    model = CNNTerrain(
        in_size=in_size,
        num_filters=num_filters,
        filter_size=filter_size,
        num_classes=num_classes,
        n_wind=n_wind,
        n_freq=n_freq,
        lr=init_learn_rate,
        learning_rate_factor=learn_drop_factor,
        reduce_lr_patience=reduce_lr_patience,
        focal_loss=focal_loss,
        focal_loss_alpha=focal_loss_alpha,
        focal_loss_gamma=focal_loss_gamma,
        scheduler=scheduler,
        dropout=dropout,
    )

    if checkpoint_path is not None:
        model = CNNTerrain.load_from_checkpoint(checkpoint_path, model=model)
        if overwrite_final_layer_dim is not None:
            model.fc = nn.Linear(model.fc.in_features, overwrite_final_layer_dim)

    exp_name = f'terrain_classification_cnn_mw_{description["mw"]}_fold_{description["fold"]}_dataset_{dataset}'
    logger = TensorBoardLogger("tb_logs", name=exp_name)

    checkpoint_folder_path = Path("checkpoints")
    trainer = L.Trainer(
        enable_progress_bar=verbose,
        enable_model_summary=verbose,
        accelerator="gpu",
        precision=32,
        logger=logger,
        log_every_n_steps=1,
        min_epochs=0,
        max_epochs=max_epochs,
        gradient_clip_val=gradient_threshold,
        val_check_interval=valid_frequency,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=valid_patience, mode="min"),
            DeviceStatsMonitor(),
            LearningRateMonitor(),
            ModelCheckpoint(
                monitor="val_loss",
                dirpath=str(checkpoint_folder_path),
                filename=f"{exp_name}-" + "{epoch:02d}-{val_loss:.6f}",
                save_top_k=1,
                save_last=True,
                mode="min",
            ),
            *custom_callbacks,
        ],
    )
    # train
    trainer.fit(model, datamodule)
    loss = trainer.validate(model, datamodule)

    if test:
        trainer.test(model, datamodule)
        out = model.test_classification
    else:
        out = model.val_classification

    out["loss"] = loss
    return out


def long_short_term_memory(
    train_data: list[ExperimentData],
    test_data: list[ExperimentData],
    lstm_par: dict,
    lstm_train_opt: dict,
    description: dict,
    custom_callbacks=None,
    test: bool = True,
) -> dict:
    if custom_callbacks is None:
        custom_callbacks = []
    # LSTM parameters
    num_classes = lstm_par["num_classes"]
    n_hidden_units = lstm_par["nHiddenUnits"]
    num_layers = lstm_par["numLayers"]
    dropout = lstm_par["dropout"]
    bidirectional = lstm_par["bidirectional"]
    convolutional = lstm_par["convolutional"]
    num_filters = lstm_par.get("numFilters", 0)
    dataset = description["dataset"]

    # Training parameters
    valid_perc = lstm_train_opt["valid_perc"]
    init_learn_rate = lstm_train_opt["init_learn_rate"]
    learn_drop_factor = lstm_train_opt["learn_drop_factor"]
    max_epochs = lstm_train_opt["max_epochs"]
    minibatch_size = lstm_train_opt["minibatch_size"]
    valid_patience = lstm_train_opt["valid_patience"]
    reduce_lr_patience = lstm_train_opt["reduce_lr_patience"]
    valid_frequency = lstm_train_opt["valid_frequency"]
    gradient_threshold = lstm_train_opt["gradient_threshold"]
    input_size = train_data["imu"].shape[-1] + train_data["pro"].shape[-1] - 10
    focal_loss = lstm_train_opt.get("focal_loss", False)
    focal_loss_alpha = lstm_train_opt.get("focal_loss_alpha", 0.25)
    focal_loss_gamma = lstm_train_opt.get("focal_loss_gamma", 2)
    num_workers = 0
    persistent_workers = True
    verbose = lstm_train_opt.get("verbose", True)

    def to_f32(x):
        return x.astype(np.float32)

    def project(data):
        imu = data["imu"]
        pro = data["pro"]
        return np.hstack([imu, pro])

    # TODO data augmentation
    augment = pp.Identity()
    train_transform = pp.Bifunctor(
        pp.Compose([project, to_f32, augment]), pp.Identity()
    )
    test_transform = pp.Bifunctor(
        pp.Compose([project, to_f32]),
        pp.Identity(),
    )

    datamodule = TemporalDataModule(
        train_data,
        test_data,
        train_transform,
        test_transform,
        valid_percent=valid_perc,
        batch_size=minibatch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    model = LSTMTerrain(
        input_size=input_size,
        hidden_size=n_hidden_units,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
        convolutional=convolutional,
        conv_num_filters=num_filters,
        num_classes=num_classes,
        lr=init_learn_rate,
        learning_rate_factor=learn_drop_factor,
        reduce_lr_patience=reduce_lr_patience,
        focal_loss=focal_loss,
        focal_loss_alpha=focal_loss_alpha,
        focal_loss_gamma=focal_loss_gamma,
    )

    exp_name = f'terrain_classification_{"c" if convolutional else ""}lstm_mw_{description["mw"]}_fold_{description["fold"]}_dataset_{dataset}'
    logger = TensorBoardLogger("tb_logs", name=exp_name)

    checkpoint_folder_path = Path("checkpoints")
    trainer = L.Trainer(
        enable_progress_bar=verbose,
        enable_model_summary=verbose,
        accelerator="gpu",
        precision=32,
        logger=logger,
        log_every_n_steps=1,
        min_epochs=0,
        max_epochs=max_epochs,
        gradient_clip_val=gradient_threshold,
        val_check_interval=valid_frequency,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=valid_patience, mode="min"),
            DeviceStatsMonitor(),
            LearningRateMonitor(),
            ModelCheckpoint(
                monitor="val_loss",
                dirpath=str(checkpoint_folder_path),
                filename=f"{exp_name}-" + "{epoch:02d}-{val_loss:.6f}",
                save_top_k=1,
                save_last=True,
                mode="min",
            ),
            *custom_callbacks,
        ],
    )
    # train
    trainer.fit(model, datamodule)
    loss = trainer.validate(model, datamodule)

    if test:
        trainer.test(model, datamodule)
        out = model.test_classification
    else:
        out = model.val_classification

    out["loss"] = loss
    return out


def support_vector_machine(
    train_dat: list[ExperimentData],
    test_dat: list[ExperimentData],
    summary: pd.DataFrame,
    n_stat_mom: int,
    svm_train_opt: dict,
    random_state: int | None = None,
) -> dict:
    """Support vector

    Args:
        train_dat (list[ExperimentData]): Train data
        test_dat (list[ExperimentData]): Test data
        summary (pd.DataFrame): Channels summary dataframe
        n_stat_mom (int): Number of statistical moments to consider
        svm_train_opt (dict): SVM training options
        random_state (int | None, optional): Random state for SVM. Defaults to None.

    Returns:
        dict: Metadata and confusion matrix across all folds
    """
    kernel_func = svm_train_opt["kernel_function"]
    poly_degree = svm_train_opt["poly_degree"]
    kernel_scale = svm_train_opt["kernel_scale"]
    box_constraint = svm_train_opt["box_constraint"]
    standardize = svm_train_opt["standardize"]
    coding = svm_train_opt["coding"]

    # Highest sampling frequency
    hf_sensor = summary["sampling_freq"].idxmax()
    # Other sensors are low frequency
    lf_sensors = tuple(sens for sens in summary.index.values if sens != hf_sensor)

    terrains = sorted(np.unique(train_dat[0][hf_sensor][:, :, 0]).tolist())

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
        skew = scst.skew(sens_data, axis=1, bias=False)
        skew[np.isnan(skew)] = 0
        # Use Pearson kurtosis
        kurt = scst.kurtosis(sens_data, axis=1, fisher=False)
        kurt[np.isnan(kurt)] = 0

        moments = np.stack([mean, std, skew, kurt], axis=2)

        return moments

    def convert_to_moments(data: ExperimentData) -> np.array:
        """Merge sensor data by applying statistical moments to them

        Args:
            data (ExperimentData): _description_

        Returns:
            Tuple[np.array]: X and y values for data
        """
        lf_moments = np.hstack([stat_moments(data[sens]) for sens in lf_sensors])
        moments = np.hstack([stat_moments(data[hf_sensor]), lf_moments])
        stat_moms = moments[:, :, :n_stat_mom]
        n_channels = stat_moms.shape[1]

        X = stat_moms.reshape(-1, n_stat_mom * n_channels, order="F")
        y = data[hf_sensor][:, 0, ch_cols["terrain"]]

        for i in range(n_stat_mom):
            idx = i * n_channels
            assert (
                stat_moms[:, :, i] == X[:, idx : idx + n_channels]
            ).all(), "Unconsistent number of channels"

        return X, y

    classification = {"pred": [], "true": [], "ftime": [], "ptime": []}

    for K_idx, (K_train, K_test) in enumerate(zip(train_dat, test_dat)):
        Xtrain_k, ytrain_k = convert_to_moments(K_train)
        xtest_k, ytest_k = convert_to_moments(K_test)

        svm = SVC(
            kernel=kernel_func,
            degree=poly_degree,
            gamma=kernel_scale,
            C=box_constraint,
            decision_function_shape="ovo",
            random_state=random_state,
        )
        ecoc = OutputCodeClassifier(
            estimator=svm,
            n_jobs=-1,
        )
        clf = make_pipeline(StandardScaler(), ecoc)

        start = time.perf_counter()
        clf.fit(Xtrain_k, ytrain_k)
        fit_time = time.perf_counter() - start

        start = time.perf_counter()
        ypred_k = clf.predict(xtest_k)
        predict_time = time.perf_counter() - start

        classification["pred"].append(ypred_k)
        classification["true"].append(ytest_k)

        print(f"Fold {K_idx} : Train {fit_time:.2f} / Test {predict_time:.2f}")

        # acc = clf.score(xtest_k, ytest_k)
        # print(f"Fold {K_idx} : SVM {acc=:.2%}")

    classification["pred"] = np.hstack(classification["pred"])
    classification["true"] = np.hstack(classification["true"])

    return classification
