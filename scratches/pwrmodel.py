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
from utils.transforms import motion_power

cwd = Path.cwd()
DATASET = os.environ.get("DATASET", "husky")  # 'husky' or 'vulpi'
if DATASET == "husky":
    csv_dir = cwd / "norlab-data"
elif DATASET == "vulpi":
    csv_dir = cwd / "data"

RANDOM_STATE = 21
HAS_PWR = True

results_dir = cwd / "results-pwr" / DATASET
results_dir.mkdir(parents=True, exist_ok=True)

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

# Add power model
if HAS_PWR:
    pro_df = terr_dfs["pro"]
    Pmotion = motion_power(pro_df)
    df = pro_df[["time", "velL", "velR", "terrain", "run_idx"]].copy()
    df.insert(1, "P_motion", Pmotion)
    terr_dfs["pro"] = df

# Set data partition parameters
NUM_CLASSES = len(np.unique(terr_dfs["imu"].terrain))
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

merged = preprocessing.merge_upsample(terr_dfs, summary, mode="last")

# Data augmentation parameters
# 0 < STRIDE < MOVING_WINDOWS
STRIDE = 0.1  # seconds
# If True, balance the classes while augmenting
# If False, imbalance the classes while augmenting
HOMOGENEOUS_AUGMENTATION = True

# SVM parameters
svm_par = {"n_stat_mom": 4}

svm_train_opt = {
    "kernel_function": "poly",
    "poly_degree": 4,
    "kernel_scale": "auto",
    "box_constraint": 100,
    "standardize": True,
    "coding": "onevsone",
}

# Model settings
BASE_MODELS = ["SVM"]
model = BASE_MODELS[0]

print(f"Training on {DATASET} with {BASE_MODELS}...")
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

    print(f"Training {model} model with {mw} seconds...")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    result_path = results_dir / f"results_{model}_mw_{mw}.npy"
    if HAS_PWR:
        result_path = result_path.with_stem(f"{result_path.stem}-pwr")
    if result_path.exists():
        print(f"Results for {model} with {mw} seconds already exist. Skipping...")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        continue

    results = models.support_vector_machine(
        aug_train,
        aug_test,
        summary,
        svm_par["n_stat_mom"],
        svm_train_opt,
        random_state=RANDOM_STATE,
    )

    # Store channels settings
    results["channels"] = columns

    # Store terrain labels
    terrains = sorted([f.stem for f in csv_dir.iterdir() if f.is_dir()])
    results["terrains"] = terrains

    np.save(result_path, results)
