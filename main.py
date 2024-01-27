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

from utils import preprocessing

cwd = Path.cwd()
data_dir = cwd / "datasets"
results_dir = cwd / "results"

# Define channels
channels = {
    "imu": {
        "sampling_freq": 50,
        "cols": {"gyrX": 1, "gyrY": 1, "gyrZ": 1, "accX": 1, "accY": 1, "accZ": 1},
    },
    "pro": {
        "sampling_freq": 15,
        "cols": {"Lvel": 1, "Rvel": 1, "Lcur": 1, "Rcur": 1},
    },
}
summary = pd.DataFrame.from_dict(channels, orient="index")

# Get recordings
terr_dfs = preprocessing.get_recordings_df(data_dir, channels)

exit(0)

# Set data partition parameters
k_fold = 5
part_window = 5  # seconds
samp_windows = [1.5, 1.6, 1.7, 1.8]  # seconds

# Data partition and sample extraction
rng_seed = 21
rng_generator = "twister"

train, test = Partition_Data(
    rec, channels, k_fold, part_window, {"seed": rng_seed, "generator": rng_generator}
)

# Data augmentation parameters
aug_sliding_window = 0.1  # seconds
aug_same = 1  # (1 or 0)

# Model settings
models = ["CNN", "LSTM", "CLSTM", "SVM"]

# CNN parameters
cnn_par = {"TimeWindow": 0.4, "TimeOvrlap": 0.2, "FilterSize": [3, 3], "numFilters": 3}

cnn_train_opt = {
    "valid_perc": 0.1,
    "init_learn_rate": 0.005,
    "learn_drop_factor": 0.1,
    "max_epochs": 150,
    "minibatch_size": 10,
    "valid_patience": 8,
    "valid_frequency": 100,
    "gradient_treshold": 6,
}


# LSTM parameters
lstm_par = {"nHiddenUnits": 15}

lstm_train_opt = {
    "valid_perc": 0.1,
    "init_learn_rate": 0.005,
    "learn_drop_factor": 0.1,
    "max_epochs": 150,
    "minibatch_size": 10,
    "valid_patience": 8,
    "valid_frequency": 100,
    "gradient_treshold": 6,
}

# CLSTM parameters
clstm_par = {"nHiddenUnits": 15, "numFilters": 5}

clstm_train_opt = {
    "valid_perc": 0.1,
    "init_learn_rate": 0.005,
    "learn_drop_factor": 0.1,
    "max_epochs": 150,
    "minibatch_size": 10,
    "valid_patience": 8,
    "valid_frequency": 100,
    "gradient_treshold": 6,
}

# SVM parameters
svm_par = {"nStatMom": 4}

svm_train_opt = {
    "kernel_function": "polynomial",
    "polynomial_order": 4,
    "kernel_scale": "auto",
    "box_constraint": 100,
    "standardize": 1,
    "coding": "onevsone",
}

# Save results name
save_name = "TrainingResults_1"

for samp_window in samp_windows:
    aug_train, aug_test = Augment_Data(
        train,
        test,
        channels,
        samp_window,
        {"sliding_window": aug_sliding_window, "same": aug_same},
    )

    print(f"Training models for a sampling window of {samp_window} seconds")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    results = {}
    for model in models:
        if model == "CNN":
            train_mcs, test_mcs = MCS_Data(aug_train, aug_test, channels, cnn_par)
            results[model] = {
                f"SampWindow_{samp_window * 1000}ms": Conv_NeuralNet(
                    train_mcs, test_mcs, cnn_par, cnn_train_opt
                )
            }
        elif model == "LSTM":
            train_ds, test_ds = DownSample_Data(aug_train, aug_test, channels)
            results[model] = {
                f"SampWindow_{samp_window * 1000}ms": LSTM_RecurrentNet(
                    train_ds, test_ds, lstm_par, lstm_train_opt
                )
            }
        elif model == "CLSTM":
            train_ds, test_ds = DownSample_Data(aug_train, aug_test, channels)
            results[model] = {
                f"SampWindow_{samp_window * 1000}ms": CLSTM_RecurrentNet(
                    train_ds, test_ds, clstm_par, clstm_train_opt
                )
            }
        elif model == "SVM":
            results[model] = {
                f"SampWindow_{samp_window * 1000}ms": SupportVectorMachine(
                    aug_train, aug_test, svm_par, svm_train_opt
                )
            }

    # Store channels settings
    results["Channels"] = channels

    # Store terrain labels
    terrains = [f.stem for f in data_dir.iterdir() if f.is_dir()]
    results["TerLabls"] = terrains

    np.save(
        results / f"{save_name}_{int(samp_window * 1000)}ms.npy",
        results,
    )
