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

from mamba_ssm.models.config_mamba import MambaConfig
from pathlib import Path

import os
import numpy as np
import pandas as pd

from utils import models, preprocessing

cwd = Path.cwd()
DATASET = os.environ.get("DATASET", "vulpi")  # 'husky' or 'vulpi'
if DATASET == "husky":
    csv_dir = cwd / "norlab-data"
elif DATASET == "vulpi":
    csv_dir = cwd / "data"
results_dir = cwd / "results"
mat_dir = cwd / "datasets"

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
terrains = sorted(terr_dfs["imu"].terrain.unique())

# Set data partition parameters
N_FOLDS = 5
PART_WINDOW = 5  # seconds
MOVING_WINDOWS = [1.5, 1.6, 1.7, 1.8]  # seconds

# Data partition and sample extraction
train_folds, test_folds = preprocessing.partition_data(
    terr_dfs,
    summary,
    PART_WINDOW,
    N_FOLDS,
    random_state=RANDOM_STATE,
)

# Data augmentation parameters
# 0 < STRIDE < MOVING_WINDOWS
STRIDE = 0.1  # seconds
# If True, balance the classes while augmenting
# If False, imbalance the classes while augmenting
HOMOGENEOUS_AUGMENTATION = True

# Mamba parameters
mamba_par = {
    "num_branches": 4,
    "norm_epsilon": 2.38e-7
}

ssm_cfg = {
    "d_state": 44,
    "d_conv": 3,
    "expand": 4,
}

mamba_cfg = MambaConfig(
    d_model=100,
    n_layer=3,
    ssm_cfg=ssm_cfg,
    rms_norm=True,
    fused_add_norm=True
)

mamba_train_opt = {
    "valid_perc": 0.1,
    "init_learn_rate": 6.33e-3,
    "learn_drop_factor": 0.27,
    "max_epochs": 150,
    "minibatch_size": 52,
    "valid_patience": 10,
    "reduce_lr_patience": 8,
    "valid_frequency": None,
    "gradient_treshold": 1,  # None to disable
    "focal_loss": True,
    "focal_loss_alpha": 0.29,
    "focal_loss_gamma": 4.87,
    "num_classes": len(terrains),
    "out_method": "max_pool" # "flatten", "max_pool", "last_state"
}

# Model settings
MODEL = "mamba"
results = {}

for mw in MOVING_WINDOWS:
    aug_train_folds, aug_test_folds = preprocessing.augment_data(
        train_folds,
        test_folds,
        summary,
        moving_window=mw,
        stride=STRIDE,
        homogeneous=HOMOGENEOUS_AUGMENTATION,
    )

    print(f"Training models for a sampling window of {mw} seconds")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    results_per_fold = []
    for k in range(N_FOLDS):
        aug_train_fold, aug_test_fold = preprocessing.prepare_data_ordering(aug_train_folds[k], aug_test_folds[k])

        out = models.mamba_network(
            aug_train_fold,
            aug_test_fold,
            mamba_par,
            mamba_train_opt,
            mamba_cfg,
            dict(mw=mw, fold=k+1, dataset=DATASET),
            random_state=RANDOM_STATE,
            test=True
        )
        results_per_fold.append(out)

    results["pred"] = np.hstack([r["pred"] for r in results_per_fold])
    results["true"] = np.hstack([r["true"] for r in results_per_fold])
    # results["conf"] = np.hstack([r["conf"] for r in results_per_fold])
    results["ftime"] = np.hstack([r["ftime"] for r in results_per_fold])
    results["ptime"] = np.hstack([r["ptime"] for r in results_per_fold])

    # Store channels settings
    results["channels"] = columns

    # Store terrain labels
    results["terrains"] = terrains

    np.save(results_dir / f"results_{MODEL}_mw_{mw}.npy", results)
