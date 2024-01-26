from pathlib import Path

import lightning as L

from utils.datamodule import VulpiDataModule
from utils.models import LSTMTerrain

if __name__ == '__main__':
    L.seed_everything(42)

    # Params
    batch_size = 32
    lr = 1e-3

    model = LSTMTerrain(lr=lr)

    datamodule = VulpiDataModule(root_dir=Path('datasets'), batch_size=batch_size)

    # logger = L.
