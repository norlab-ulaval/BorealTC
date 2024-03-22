"""
Proprioception Is All You Need: Terrain Classification for Boreal Forests
Damien LaRocque*, William Guimont-Martin, David-Alexandre Duclos, Philippe Giguère, Francois Pomerleau
---
This script was inspired by the MAIN.m script in the T_DEEP repository from Ph0bi0 : https://github.com/Ph0bi0/T_DEEP
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils import models, preprocessing

cwd = Path.cwd()
DATASET = os.environ.get("DATASET", "husky")  # 'husky' or 'vulpi'
if DATASET == "husky":
    csv_dir = cwd / "norlab-data"
elif DATASET == "vulpi":
    csv_dir = cwd / "data"

mat_dir = cwd / "datasets"
results_dir = cwd / "results" / DATASET
results_dir.mkdir(parents=True, exist_ok=True)

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
terrains = sorted(terr_dfs["imu"].terrain.unique().tolist())

# Set data partition parameters
NUM_CLASSES = len(terrains)
N_FOLDS = 5
PART_WINDOW = 5  # seconds
MOVING_WINDOWS = [1.5, 1.6, 1.7, 1.8]  # seconds

# Data partition and sample extraction
train, test = preprocessing.partition_data(
    terr_dfs,
    summary,
    PART_WINDOW,
    N_FOLDS,
    random_state=RANDOM_STATE,
)

# merged = preprocessing.merge_upsample(terr_dfs, summary, mode="last")

# Data augmentation parameters
# 0 < STRIDE < MOVING_WINDOWS
STRIDE = 0.1  # seconds
# If True, balance the classes while augmenting
# If False, imbalance the classes while augmenting
HOMOGENEOUS_AUGMENTATION = True

# CNN parameters
cnn_par = {
    "num_classes": NUM_CLASSES,
    "time_window": 0.4,
    "time_overlap": 0.2,
    "filter_size": [3, 3],
    "num_filters": 3,
}

cnn_train_opt = {
    "valid_perc": 0.1,
    "init_learn_rate": 0.005,
    "learn_drop_factor": 0.1,
    "max_epochs": 150,
    "minibatch_size": 10,
    "valid_patience": 8,
    "reduce_lr_patience": 4,
    "valid_frequency": 100,
    "gradient_threshold": 6,  # None to disable
    "verbose": False,
}

# Model settings
MODEL = "CNN"

print(f"Training on {DATASET} with {MODEL}...")
for mw in MOVING_WINDOWS:
    aug_train, aug_test = preprocessing.augment_data(
        train,
        test,
        summary,
        moving_window=mw,
        stride=STRIDE,
        homogeneous=HOMOGENEOUS_AUGMENTATION,
    )

    print(f"Training models for a sampling window of {mw} seconds")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print(f"Training {MODEL} model with {mw} seconds...")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    result_path = results_dir / f"results_{MODEL}_mw_{mw}.npy"
    if result_path.exists():
        print(f"Results for {MODEL} with {mw} seconds already exist. Skipping...")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        continue

    results = {}

    (
        train_mcs_folds,
        test_mcs_folds,
    ) = preprocessing.apply_multichannel_spectogram(
        aug_train,
        aug_test,
        summary,
        mw,
        cnn_par["time_window"],
        cnn_par["time_overlap"],
        hamming=False,
    )
    results_per_fold = []
    for k in range(N_FOLDS):
        train_mcs, test_mcs = train_mcs_folds[k], test_mcs_folds[k]
        out = models.convolutional_neural_network(
            train_mcs,
            test_mcs,
            cnn_par,
            cnn_train_opt,
            dict(mw=mw, fold=k + 1, dataset=DATASET),
            random_state=RANDOM_STATE,
        )
        results_per_fold.append(out)

    results["pred"] = np.hstack([r["pred"] for r in results_per_fold])
    results["true"] = np.hstack([r["true"] for r in results_per_fold])
    results["conf"] = np.vstack([r["conf"] for r in results_per_fold])
    results["ftime"] = np.hstack([r["ftime"] for r in results_per_fold])
    results["ptime"] = np.hstack([r["ptime"] for r in results_per_fold])

    # Store channels settings
    results["channels"] = columns

    # Store terrain labels
    results["terrains"] = terrains

    np.save(result_path, results)
