from __future__ import annotations

import pathlib
import time
from typing import TYPE_CHECKING

import lightning as L
import numpy as np
import pandas as pd
import pipeline as pp
import scipy.stats as scst
import torch
import torch.nn as nn
import torchmetrics
import torchvision as tv
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint, EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.multiclass import OutputCodeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils.constants import ch_cols
from utils.datamodule import MCSDataModule

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
            filter_size: int | (int, int),
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
        self.in_layer = nn.Conv2d(in_size, in_size, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(in_size)
        self.conv2d1 = nn.Conv2d(
            in_size, num_filters, kernel_size=filter_size, padding="same"
        )
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_filters * self.n_wind * self.n_freq, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights) if class_weights else None
        )

        self._train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)
        self._val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)
        self._test_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)

        self._val_classifications = [{"pred": [], "true": [], "ftime": [], "ptime": []}]
        self._test_classifications = [{"pred": [], "true": [], "ftime": [], "ptime": []}]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,
        #                                                       lr_lambda=lambda epoch: self.learning_rate_factor,
        #                                                       verbose=True)
        # TODO try ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.reduce_lr_patience,
                                                               factor=self.learning_rate_factor, verbose=True)
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                monitor="val_loss",
            )
        )

    @property
    def val_classification(self):
        return self._val_classifications[-1]

    @property
    def test_classification(self):
        return self._test_classifications[-1]

    def forward(self, x):
        x = self.in_layer(x)
        x = self.batch_norm(x)
        x = self.conv2d1(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x

    def loss(self, y, target):
        if self.focal_loss:
            return tv.ops.sigmoid_focal_loss(
                y, target, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma
            )
        return self.ce_loss(y, target)

    def training_step(self, batch, batch_idx):
        x, target = batch
        y = self(x)
        loss = self.loss(y, target)
        acc = self._train_accuracy(y, target)

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_accuracy", acc, on_step=True)

        return loss

    def on_train_epoch_end(self):
        self.log("train_accuracy_epoch", self._train_accuracy.compute(), prog_bar=True, on_epoch=True)
        self._train_accuracy.reset()

    def validation_step(self, val_batch, batch_idx):
        x, target = val_batch
        y = self(x)
        loss = self.loss(y, target)
        acc = self._val_accuracy(y, target)

        self.log("val_loss", loss, on_step=True)
        self.log("val_accuracy", acc, on_step=True)

        pred_classes = torch.argmax(y, dim=1)
        self._val_classifications[-1]["pred"].append(
            pred_classes.detach().cpu().numpy()
        )
        self._val_classifications[-1]["true"].append(target.detach().cpu().numpy())

        return loss

    def on_validation_epoch_end(self):
        self.log("val_accuracy_epoch", self._val_accuracy.compute(), prog_bar=True, on_epoch=True)
        self._val_accuracy.reset()

    def on_validation_end(self):
        self._val_classifications[-1]["pred"] = np.hstack(
            self._val_classifications[-1]["pred"]
        )
        self._val_classifications[-1]["true"] = np.hstack(
            self._val_classifications[-1]["true"]
        )
        self._val_classifications.append(
            {"pred": [], "true": [], "ftime": [], "ptime": []}
        )

    def test_step(self, test_batch, batch_idx):
        x, target = test_batch
        y = self(x)
        loss = self.loss(y, target)
        acc = self._test_accuracy(y, target)

        self.log("test_loss", loss, on_step=True)
        self.log("test_accuracy", acc, on_step=True)

        pred_classes = torch.argmax(y, dim=1)
        self._test_classifications[-1]["pred"].append(pred_classes.detach().cpu().numpy())
        self._test_classifications[-1]["true"].append(target.detach().cpu().numpy())

        return loss

    def on_test_epoch_end(self):
        self.log("test_accuracy_epoch", self._test_accuracy.compute(), prog_bar=True, on_epoch=True)
        self._test_accuracy.reset()

    def on_test_end(self):
        self._test_classifications[-1]["pred"] = np.hstack(self._test_classifications[-1]["pred"])
        self._test_classifications[-1]["true"] = np.hstack(self._test_classifications[-1]["true"])
        self._test_classifications.append({"pred": [], "true": [], "ftime": [], "ptime": []})


def convolutional_neural_network(
        train_data: list[ExperimentData],
        test_data: list[ExperimentData],
        cnn_par: dict,
        cnn_train_opt: dict,
        description: dict,
) -> dict:
    # CNN parameters
    filter_size = cnn_par['filter_size']
    num_filters = cnn_par['num_filters']

    # Training parameters
    valid_perc = cnn_train_opt['valid_perc']
    init_learn_rate = cnn_train_opt['init_learn_rate']
    learn_drop_factor = cnn_train_opt['learn_drop_factor']
    max_epochs = cnn_train_opt['max_epochs']
    minibatch_size = cnn_train_opt['minibatch_size']
    valid_patience = cnn_train_opt['valid_patience']
    reduce_lr_patience = cnn_train_opt['reduce_lr_patience']
    valid_frequency = cnn_train_opt['valid_frequency']
    gradient_treshold = cnn_train_opt['gradient_treshold']
    _, n_freq, n_wind, in_size = train_data['data'].shape
    num_workers = 8
    persistent_workers = True

    def to_f32(x):
        return x.astype(np.float32)

    def transpose(x):
        return np.transpose(x, (2, 0, 1))

    # TODO move spectro generation here
    augment = pp.Identity()
    to_mcs = pp.Identity()
    train_transform = pp.Bifunctor(
        pp.Compose([to_f32, transpose, augment, to_mcs]),
        pp.Identity(),
    )
    test_transform = pp.Bifunctor(
        pp.Compose([to_f32, transpose, to_mcs]),
        pp.Identity(),
    )
    datamodule = MCSDataModule(
        train_data,
        test_data,
        train_transform,
        test_transform,
        valid_percent=valid_perc,
        batch_size=minibatch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers)

    model = CNNTerrain(
        in_size=in_size,
        num_filters=num_filters,
        filter_size=filter_size,
        num_classes=4,
        n_wind=n_wind,
        n_freq=n_freq,
        lr=init_learn_rate,
        learning_rate_factor=learn_drop_factor,
        reduce_lr_patience=reduce_lr_patience)

    exp_name = (
        f'terrain_classification_mw_{description["mw"]}_fold_{description["fold"]}'
    )
    logger = TensorBoardLogger("tb_logs", name=exp_name)

    checkpoint_folder_path = pathlib.Path('checkpoints')
    trainer = L.Trainer(accelerator='gpu', precision=32, logger=logger, log_every_n_steps=1,
                        min_epochs=0, max_epochs=max_epochs,
                        gradient_clip_val=gradient_treshold,
                        val_check_interval=valid_frequency,
                        callbacks=[
                            EarlyStopping(monitor='val_loss', patience=valid_patience, mode='min'),
                            DeviceStatsMonitor(),
                            LearningRateMonitor(),
                            ModelCheckpoint(
                                monitor='val_loss',
                                dirpath=str(checkpoint_folder_path),
                                filename=f'{exp_name}-' + '{epoch:02d}-{val_loss:.6f}',
                                save_top_k=1,
                                save_last=True,
                                mode='min',
                            )])
    # train
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)

    return model.test_classification


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
                    stat_moms[:, :, i] == X[:, idx: idx + n_channels]
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
