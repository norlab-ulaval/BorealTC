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

# Set data partition parameters
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

# CNN parameters
cnn_par = {
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
    "gradient_treshold": 6,  # None to disable
}

# Model settings
MODEL = "CNN"
results = {}
# BASE_MODELS = ["CNN", "SVM"]

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

    (
        train_mcs_folds,
        test_mcs_folds,
    ) = preprocessing.apply_multichannel_spectogram(
        aug_train,
        aug_test,
        summary,
        cnn_par["time_window"],
        cnn_par["time_overlap"],
    )
    results_per_fold = []
    for k in range(N_FOLDS):
        train_mcs, test_mcs = train_mcs_folds[k], test_mcs_folds[k]
        out = models.convolutional_neural_network(
            train_mcs, test_mcs, cnn_par, cnn_train_opt, dict(mw=mw, fold=k + 1)
        )
        results_per_fold.append(out)

    results["pred"] = np.hstack([r["pred"] for r in results_per_fold])
    results["true"] = np.hstack([r["true"] for r in results_per_fold])
    results["conf"] = np.hstack([r["conf"] for r in results_per_fold])
    results["ftime"] = np.hstack([r["ftime"] for r in results_per_fold])
    results["ptime"] = np.hstack([r["ptime"] for r in results_per_fold])

    # results[model] = {
    #     f"{samp_window * 1000}ms": Conv_NeuralNet(
    #         train_mcs, test_mcs, cnn_par, cnn_train_opt
    #     )
    # }

    # Store channels settings
    results["channels"] = columns

    # Store terrain labels
    terrains = sorted([f.stem for f in csv_dir.iterdir() if f.is_dir()])
    results["terrains"] = terrains

    np.save(results_dir / f"results_{MODEL}_mw_{mw}.npy", results)
