import pathlib
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, DeviceStatsMonitor, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from utils.datamodule import VulpiDataModule, to_spectrogram
from utils.models import CNNTerrain

if __name__ == '__main__':
    L.seed_everything(42)

    # Params
    batch_size = 32
    lr = 1e-3
    n_wind = 16
    n_freq = 128

    model = CNNTerrain(in_size=10, num_filters=3, filter_size=3, num_classes=4, n_wind=n_wind, n_freq=n_freq, lr=lr)

    datamodule = VulpiDataModule(root_dir=Path('datasets'), batch_size=batch_size, transform=to_spectrogram)

    # logger = L.
    # TODO gradient clipping
    logger = TensorBoardLogger('tb_logs', name='terrain_classification')

    checkpoint_folder_path = pathlib.Path('checkpoints')
    trainer = L.Trainer(accelerator='gpu', precision=32, logger=logger, log_every_n_steps=1,
                        min_epochs=0, max_epochs=1000,
                        gradient_clip_val=6,  # TODO validate that this is necessary
                        callbacks=[EarlyStopping('val_loss', patience=15, verbose=True),
                                   DeviceStatsMonitor(),
                                   LearningRateMonitor(),
                                   ModelCheckpoint(
                                       monitor='val_loss',
                                       dirpath=str(checkpoint_folder_path),
                                       filename='terrain-{epoch:02d}-{val_loss:.6f}',
                                       save_top_k=1,
                                       save_last=True,
                                       mode='min',
                                   )])
    # train
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
