from pathlib import Path
from typing import Optional, Callable

import lightning as L
from torch.utils.data import random_split, DataLoader

from utils.dataset import RawVulpiDataset, MCSDataset


class VulpiDataModule(L.LightningDataModule):
    def __init__(self, root_dir: Path, transform: Optional[Callable] = None, batch_size: int = 32, split: float = 0.8,
                 num_workers: int = 8, persistent_workers: bool = True):
        super().__init__()
        self.dataset = RawVulpiDataset(root_dir, transform)
        self.train_split, self.val_split = self._split(self.dataset, split=split)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    @staticmethod
    def _split(dataset, split: float):
        return random_split(dataset, [split, 1 - split])

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=True, drop_last=False, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_split, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=False, drop_last=False, persistent_workers=self.persistent_workers)


class MCSDataModule(L.LightningDataModule):
    def __init__(self, train_mcs, test_mcs, train_transform: Optional[Callable] = None,
                 test_transform: Optional[Callable] = None, batch_size: int = 32, num_workers: int = 8,
                 persistent_workers: bool = True):
        super().__init__()
        self.train_dataset = MCSDataset(train_mcs, train_transform)
        self.test_dataset = MCSDataset(test_mcs, test_transform)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=True, drop_last=False, persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=False, drop_last=False, persistent_workers=self.persistent_workers)
