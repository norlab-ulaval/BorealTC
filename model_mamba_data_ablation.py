"""
Proprioception Is All You Need: Terrain Classification for Boreal Forests
Damien LaRocque*, William Guimont-Martin, David-Alexandre Duclos, Philippe Giguère, Francois Pomerleau
---
This script was inspired by the MAIN.m script in the T_DEEP repository from Ph0bi0 : https://github.com/Ph0bi0/T_DEEP
"""

from pathlib import Path

import os
import numpy as np
import pandas as pd

from utils import models, preprocessing

cwd = Path.cwd()

DATASET = os.environ.get("DATASET", "vulpi")  # 'husky' or 'vulpi' or 'combined'
COMBINED_PRED_TYPE = os.environ.get(
    "COMBINED_PRED_TYPE", "class"
)  # 'class' or 'dataset'
CHECKPOINT = os.environ.get("CHECKPOINT", None)

if DATASET == "husky":
    csv_dir = cwd / "data" / "borealtc"
elif DATASET == "vulpi":
    csv_dir = cwd / "data" / "vulpi"
elif DATASET == "combined":
    csv_dir = dict(vulpi=cwd / "data" / "vulpi", husky=cwd / "data" / "borealtc")

results_dir = cwd / "results"
mat_dir = cwd / "data"

if CHECKPOINT is not None:
    CHECKPOINT = cwd / "checkpoints" / CHECKPOINT

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
if DATASET == "combined":
    summary = {}

    for key in csv_dir.keys():
        summary[key] = pd.DataFrame({"columns": pd.Series(columns)})
else:
    summary = pd.DataFrame({"columns": pd.Series(columns)})

# Get recordings
if DATASET == "combined":
    terr_dfs = {}
    terrains = []

    terr_df_husky = preprocessing.get_recordings(csv_dir["husky"], summary["husky"])
    terr_df_vulpi = preprocessing.get_recordings(csv_dir["vulpi"], summary["vulpi"])

    terr_dfs["husky"] = terr_df_husky
    terr_dfs["vulpi"] = terr_df_vulpi

    if COMBINED_PRED_TYPE == "class":
        for key in csv_dir.keys():
            terrains += sorted(terr_dfs[key]["imu"].terrain.unique())
    elif COMBINED_PRED_TYPE == "dataset":
        terrains = list(csv_dir.keys())
else:
    terr_dfs = preprocessing.get_recordings(csv_dir, summary)
    terrains = sorted(terr_dfs["imu"].terrain.unique())

# Set data partition parameters
N_FOLDS = 5
PART_WINDOW = 5  # seconds
# MOVING_WINDOWS = [1.5, 1.6, 1.7, 1.8]  # seconds
MOVING_WINDOWS = [1.7]  # seconds

# Data partition and sample extraction
if DATASET == "combined":
    train_folds = {}
    test_folds = {}

    for key in csv_dir.keys():
        _train_folds, _test_folds = preprocessing.partition_data(
            terr_dfs[key],
            summary[key],
            PART_WINDOW,
            N_FOLDS,
            random_state=RANDOM_STATE,
            ablation=True,
        )
        train_folds[key] = _train_folds
        test_folds[key] = _test_folds
else:
    train_folds, test_folds = preprocessing.partition_data(
        terr_dfs,
        summary,
        PART_WINDOW,
        N_FOLDS,
        random_state=RANDOM_STATE,
        ablation=True,
    )

# Data augmentation parameters
# 0 < STRIDE < MOVING_WINDOWS
STRIDE = 0.1  # seconds
# If True, balance the classes while augmenting
# If False, imbalance the classes while augmenting
HOMOGENEOUS_AUGMENTATION = True

# Mamba parameters
mamba_par = {"d_model_imu": 32, "d_model_pro": 8, "norm_epsilon": 6.3e-6}

ssm_cfg_imu = {"d_state": 16, "d_conv": 4, "expand": 4}

ssm_cfg_pro = {"d_state": 16, "d_conv": 3, "expand": 6}

mamba_train_opt = {
    "valid_perc": 0.1,
    "init_learn_rate": 1.5e-3,
    "learn_drop_factor": 0.25,
    "max_epochs": 60,
    "minibatch_size": 16,
    "valid_patience": 8,
    "reduce_lr_patience": 4,
    "valid_frequency": None,
    "gradient_treshold": None,  # None to disable
    "focal_loss": True,
    "focal_loss_alpha": 0.75,
    "focal_loss_gamma": 2.25,
    "num_classes": len(terrains),
    "out_method": "last_state",  # "max_pool", "last_state"
}

# Model settings
MODEL = "mamba"
results = {}

for mw in MOVING_WINDOWS:
    if DATASET == "combined":
        aug_train_folds = {}
        aug_test_folds = {}

        for key in csv_dir.keys():
            _aug_train_folds, _aug_test_folds = preprocessing.augment_data_ablation(
                train_folds[key],
                test_folds[key],
                summary[key],
                moving_window=mw,
                stride=STRIDE,
                homogeneous=HOMOGENEOUS_AUGMENTATION,
            )
            aug_train_folds[key] = _aug_train_folds
            aug_test_folds[key] = _aug_test_folds
    else:
        aug_train_folds, aug_test_folds = preprocessing.augment_data_ablation(
            train_folds,
            test_folds,
            summary,
            moving_window=mw,
            stride=STRIDE,
            homogeneous=HOMOGENEOUS_AUGMENTATION,
        )

    print(f"Training models for a sampling window of {mw} seconds")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    for k in range(N_FOLDS):
        if DATASET == "combined":
            aug_train_fold = {}
            aug_test_fold = {}

            for key in csv_dir.keys():
                _aug_train_fold, _aug_test_fold = preprocessing.cleanup_data_ablation(
                    aug_train_folds[key][k], aug_test_folds[key][k]
                )
                _aug_train_fold, _aug_test_fold = preprocessing.normalize_data_ablation(
                    _aug_train_fold, _aug_test_fold
                )

                aug_train_fold[key] = _aug_train_fold
                aug_test_fold[key] = _aug_test_fold

            if COMBINED_PRED_TYPE == "class":
                num_classes_vulpi = len(np.unique(aug_train_fold["vulpi"][0]["labels"]))

                for _k in range(N_FOLDS):
                    aug_train_fold["husky"][_k]["labels"] += num_classes_vulpi
                aug_test_fold["husky"]["labels"] += num_classes_vulpi
            elif COMBINED_PRED_TYPE == "dataset":
                for _k in range(N_FOLDS):
                    aug_train_fold["vulpi"][_k]["labels"] = np.full_like(
                        aug_train_fold["vulpi"][_k]["labels"], 0
                    )
                    aug_train_fold["husky"][_k]["labels"] = np.full_like(
                        aug_train_fold["husky"][_k]["labels"], 1
                    )

                aug_test_fold["vulpi"]["labels"] = np.full_like(
                    aug_test_fold["vulpi"]["labels"], 0
                )
                aug_test_fold["husky"]["labels"] = np.full_like(
                    aug_test_fold["husky"]["labels"], 1
                )

            aug_train_folds["vulpi"][k] = aug_train_fold["vulpi"]
            aug_train_folds["husky"][k] = aug_train_fold["husky"]
            aug_test_folds["vulpi"][k] = aug_test_fold["vulpi"]
            aug_test_folds["husky"][k] = aug_test_fold["husky"]
        else:
            aug_train_fold, aug_test_fold = preprocessing.cleanup_data_ablation(
                aug_train_folds[k], aug_test_folds[k]
            )
            aug_train_fold, aug_test_fold = preprocessing.normalize_data_ablation(
                aug_train_fold, aug_test_fold
            )

            aug_train_folds[k] = aug_train_fold
            aug_test_folds[k] = aug_test_fold

for mw in MOVING_WINDOWS:
    for _k in reversed(range(N_FOLDS)):  # subsample sizes
        results_per_fold = []

        for k in range(N_FOLDS):  # kfolds
            if DATASET == "combined":
                aug_train_fold = dict(
                    vulpi=aug_train_folds["vulpi"][k][_k],
                    husky=aug_train_folds["husky"][k][_k],
                )
                aug_test_fold = dict(
                    vulpi=aug_test_folds["vulpi"][k], husky=aug_test_folds["husky"][k]
                )
            else:
                aug_train_fold = aug_train_folds[k][_k]
                aug_test_fold = aug_test_folds[k]

            out = models.mamba_network(
                aug_train_fold,
                aug_test_fold,
                mamba_par,
                mamba_train_opt,
                ssm_cfg_imu,
                ssm_cfg_pro,
                dict(mw=mw, fold=k + 1, dataset=DATASET),
                random_state=RANDOM_STATE,
                test=True,
                checkpoint=CHECKPOINT,
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

        np.save(results_dir / f"results_split_{_k+1}_{MODEL}_mw_{mw}.npy", results)
