from pathlib import Path
from typing import Optional, Callable

import lightning as L
from torch.utils.data import random_split, DataLoader

from utils.dataset import VulpiDataset


class VulpiDataModule(L.LightningDataModule):
    def __init__(self, root_dir: Path, transform: Optional[Callable] = None, batch_size: int = 32, split: float = 0.8,
                 num_workers: int = 8):
        super().__init__()
        self.dataset = VulpiDataset(root_dir, transform)
        self.train_split, self.val_split = self._split(self.dataset, split=split)
        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def _split(dataset, split: float):
        return random_split(dataset, [split, 1 - split])

    def train_dataloader(self):
        return DataLoader(self.train_split, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_split, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=True, shuffle=False, drop_last=True)
