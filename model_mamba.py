"""
AUTHOR INFORMATION AND SCRIPT OVERVIEW, V1.0, 12/2020
Author:________________________________________Fabio Vulpi (Github: Ph0bi0)

                                 PhD student at Polytechnic of Bari (Italy)
                     Researcher at National Research Council of Italy (CNR)

This is the main script to train and test deep terrain classification
models:
- Convolutional Neural Network (CNN)
- Long Short-Term Memory recurrent neural network (LSTM)
- Convolutional Long Short-Term Memory recurrent neural network (CLSTM)

The script also uses a state of the art Support Vector Machine (SVM) as
benchmark
-------------------------------------------------------------------------
"""

from pathlib import Path

import numpy as np
import pandas as pd

from utils import models, preprocessing

cwd = Path.cwd()
mat_dir = cwd / "datasets"
csv_dir = cwd / "data"
results_dir = cwd / "results"
csv_dir = cwd / "norlab-data"

RANDOM_STATE = 21

# Define channels
columns = {
    "imu": {
        "wx": True,
        "wy": True,
        "wz": True,
        "ax": True,
        "ay": True,
        "az": True,
    },
    "pro": {
        "velL": True,
        "velR": True,
        "curL": True,
        "curR": True,
    },
}
summary = pd.DataFrame({"columns": pd.Series(columns)})

# Get recordings
terr_dfs = preprocessing.get_recordings(csv_dir, summary)

# # Set data partition parameters
N_FOLDS = None
PART_WINDOW = 5  # seconds

# Data partition and sample extraction
train, test = preprocessing.partition_data(
    terr_dfs,
    summary,
    PART_WINDOW,
    N_FOLDS,
    random_state=RANDOM_STATE,
)

# Mamba parameters
mamba_par = {"model_dim": 16, "state_factor": 16, "conv_width": 4, "expand_factor": 2}

mamba_train_opt = {
    "valid_perc": 0.1,
    "init_learn_rate": 0.005,
    "learn_drop_factor": 0.1,
    "max_epochs": 150,
    "minibatch_size": 10,
    "valid_patience": 8,
    "reduce_lr_patience": 4,
    "valid_frequency": 100,
    "gradient_treshold": 6,  # None to disable
}

# Model settings
MODEL = "mamba"

results = models.mamba_network(train, test, mamba_par, mamba_train_opt)

# Store channels settings
results["channels"] = columns

# Store terrain labels
terrains = sorted([f.stem for f in csv_dir.iterdir() if f.is_dir()])
results["terrains"] = terrains

data_name = 'norlab' if csv_dir == 'norlab-data' else 'vulpi'

np.save(results_dir / f"results_{MODEL}_part_{PART_WINDOW}_data_{data_name}.npy", results)
