from pathlib import Path
from typing import Optional, Callable

import lightning as L
from lightning.pytorch.utilities import CombinedLoader
import numpy as np
from torch.utils.data import random_split, DataLoader

from utils.dataset import RawVulpiDataset, MCSDataset, TemporalDataset, MambaDataset


def _split(dataset, valid_percent: float):
    num_sample = len(dataset)
    train_percent = 1 - valid_percent
    num_train = int(np.ceil(num_sample * train_percent))
    num_val = int(np.floor(num_sample * valid_percent))
    return random_split(dataset, [num_train, num_val])


class VulpiDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: Path,
        transform: Optional[Callable] = None,
        batch_size: int = 32,
        split: float = 0.8,
        num_workers: int = 8,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.dataset = RawVulpiDataset(root_dir, transform)
        self.train_split, self.val_split = _split(self.dataset, valid_percent=split)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )


class CustomDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_type,
        train_temporal,
        test_temporal,
        train_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        valid_percent: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 8,
        persistent_workers: bool = True,
    ):
        super().__init__()
        train_dataset = dataset_type(train_temporal, train_transform)
        self.train_dataset, self.val_dataset = _split(train_dataset, valid_percent=valid_percent)
        self.test_dataset = dataset_type(test_temporal, test_transform)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            persistent_workers=self.persistent_workers,
        )


class TemporalDataModule(CustomDataModule):
    def __init__(
        self,
        train_temporal,
        test_temporal,
        train_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        valid_percent: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 8,
        persistent_workers: bool = True,
    ):
        super().__init__(
            TemporalDataset,
            train_temporal,
            test_temporal,
            train_transform,
            test_transform,
            valid_percent,
            batch_size,
            num_workers,
            persistent_workers,
        )


class MCSDataModule(CustomDataModule):
    def __init__(
        self,
        train_temporal,
        test_temporal,
        train_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        valid_percent: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 8,
        persistent_workers: bool = True,
    ):
        super().__init__(
            MCSDataset,
            train_temporal,
            test_temporal,
            train_transform,
            test_transform,
            valid_percent,
            batch_size,
            num_workers,
            persistent_workers,
        )


class MambaDataModule(CustomDataModule):
    def __init__(
        self,
        train_temporal,
        test_temporal,
        train_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        valid_percent: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 8,
        persistent_workers: bool = True,
    ):
        super().__init__(
            MambaDataset,
            train_temporal,
            test_temporal,
            train_transform,
            test_transform,
            valid_percent,
            batch_size,
            num_workers,
            persistent_workers,
        )


class MambaDataModuleCombined(L.LightningDataModule):
    def __init__(
        self,
        train_temporal_vulpi,
        test_temporal_vulpi,
        train_temporal_husky,
        test_temporal_husky,
        train_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        valid_percent: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 8,
        persistent_workers: bool = True
    ):
        super().__init__()

        train_dataset_vulpi = MambaDataset(train_temporal_vulpi, train_transform)
        train_dataset_husky = MambaDataset(train_temporal_husky, train_transform)

        self.train_dataset_vulpi, self.val_dataset_vulpi = _split(train_dataset_vulpi, valid_percent=valid_percent)
        self.train_dataset_husky, self.val_dataset_husky = _split(train_dataset_husky, valid_percent=valid_percent)

        self.test_dataset_vulpi = MambaDataset(test_temporal_vulpi, test_transform)
        self.test_dataset_husky = MambaDataset(test_temporal_husky, test_transform)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def train_dataloader(self):
        return CombinedLoader(
            [
                DataLoader(
                    self.train_dataset_vulpi,
                    batch_size=self.batch_size // 2,
                    num_workers=self.num_workers // 2,
                    pin_memory=True,
                    shuffle=True,
                    drop_last=False,
                    persistent_workers=self.persistent_workers,
                ), DataLoader(
                    self.train_dataset_husky,
                    batch_size=self.batch_size // 2,
                    num_workers=self.num_workers // 2,
                    pin_memory=True,
                    shuffle=True,
                    drop_last=False,
                    persistent_workers=self.persistent_workers,
                )
            ], mode="min_size"
        )

    def val_dataloader(self):
        return CombinedLoader(
            [
                DataLoader(
                    self.val_dataset_vulpi,
                    batch_size=self.batch_size // 2,
                    num_workers=self.num_workers // 2,
                    pin_memory=True,
                    shuffle=False,
                    drop_last=False,
                    persistent_workers=self.persistent_workers,
                ), DataLoader(
                    self.val_dataset_husky,
                    batch_size=self.batch_size // 2,
                    num_workers=self.num_workers // 2,
                    pin_memory=True,
                    shuffle=False,
                    drop_last=False,
                    persistent_workers=self.persistent_workers,
                )
            ], mode="min_size" # mode="sequential"
        )

    def test_dataloader(self):
        return CombinedLoader(
            [
                DataLoader(
                    self.test_dataset_vulpi,
                    batch_size=self.batch_size // 2,
                    num_workers=self.num_workers // 2,
                    pin_memory=True,
                    shuffle=False,
                    drop_last=False,
                    persistent_workers=self.persistent_workers,
                ), DataLoader(
                    self.test_dataset_husky,
                    batch_size=self.batch_size // 2,
                    num_workers=self.num_workers // 2,
                    pin_memory=True,
                    shuffle=False,
                    drop_last=False,
                    persistent_workers=self.persistent_workers,
                )
            ], mode="min_size" # mode="sequential"
        )
