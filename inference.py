import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils import mixed

cwd = Path.cwd()
csv_dir = cwd / "norlab-data" / "MIXED"

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
terr_dfs = mixed.mx_get_recordings(csv_dir, summary)

# Set data partition parameters
NUM_CLASSES = len(np.unique(terr_dfs["imu"].terrain))
N_FOLDS = 5
MOVING_WINDOWS = [1.5, 1.6, 1.7, 1.8]  # seconds

# Model settings
BASE_MODELS = ["SVM", "CNN", "LSTM", "CLSTM"]

print(f"Infering on MIXED run with {BASE_MODELS}...")
for mw in MOVING_WINDOWS:
    # Data partition and sample extraction
    partitions = mixed.mx_partition_data(
        terr_dfs,
        summary,
        mw,
        N_FOLDS,
        random_state=RANDOM_STATE,
    )

    print(f"Training models for a sampling window of {mw} seconds")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    for model in BASE_MODELS:
        print(f"Training {model} model with {mw} seconds...")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        result_path = results_dir / f"results_{model}_mw_{mw}.npy"
        if result_path.exists():
            print(f"Results for {model} with {mw} seconds already exist. Skipping...")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            continue

        results = {}

        if model == "CNN":
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

            # results[model] = {
            #     f"{samp_window * 1000}ms": Conv_NeuralNet(
            #         train_mcs, test_mcs, cnn_par, cnn_train_opt
            #     )
            # }
        elif model == "LSTM":
            train_ds_folds, test_ds_folds = preprocessing.downsample_data(
                aug_train,
                aug_test,
                summary,
            )
            results_per_fold = []
            for k in range(N_FOLDS):
                train_ds, test_ds = train_ds_folds[k], test_ds_folds[k]
                out = models.long_short_term_memory(
                    train_ds,
                    test_ds,
                    lstm_par,
                    lstm_train_opt,
                    dict(mw=mw, fold=k + 1, dataset=DATASET),
                )
                results_per_fold.append(out)

            results["pred"] = np.hstack([r["pred"] for r in results_per_fold])
            results["true"] = np.hstack([r["true"] for r in results_per_fold])
            results["conf"] = np.vstack([r["conf"] for r in results_per_fold])
            results["ftime"] = np.hstack([r["ftime"] for r in results_per_fold])
            results["ptime"] = np.hstack([r["ptime"] for r in results_per_fold])
        elif model == "CLSTM":
            train_ds_folds, test_ds_folds = preprocessing.downsample_data(
                aug_train,
                aug_test,
                summary,
            )
            results_per_fold = []
            for k in range(N_FOLDS):
                train_ds, test_ds = train_ds_folds[k], test_ds_folds[k]
                out = models.long_short_term_memory(
                    train_ds,
                    test_ds,
                    clstm_par,
                    clstm_train_opt,
                    dict(mw=mw, fold=k + 1, dataset=DATASET),
                )
                results_per_fold.append(out)

            results["pred"] = np.hstack([r["pred"] for r in results_per_fold])
            results["true"] = np.hstack([r["true"] for r in results_per_fold])
            results["conf"] = np.vstack([r["conf"] for r in results_per_fold])
            results["ftime"] = np.hstack([r["ftime"] for r in results_per_fold])
            results["ptime"] = np.hstack([r["ptime"] for r in results_per_fold])
        elif model == "SVM":
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
