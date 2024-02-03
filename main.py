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

import pandas as pd

from utils import models, preprocessing

cwd = Path.cwd()
mat_dir = cwd / "datasets"
csv_dir = cwd / "data"
results_dir = cwd / "results"

# Define channels
columns = {
    "imu": {
        "gyrX": True,
        "gyrY": True,
        "gyrZ": True,
        "accX": True,
        "accY": True,
        "accZ": True,
    },
    "pro": {
        "Lvel": True,
        "Rvel": True,
        "Lcur": True,
        "Rcur": True,
    },
}
summary = pd.DataFrame({"columns": pd.Series(columns)})

# Get recordings
terr_dfs = preprocessing.get_recordings_csv(csv_dir, summary)


# Set data partition parameters
N_FOLDS = 5
PART_WINDOW = 5  # seconds
MOVING_WINDOWS = [1.5, 1.6, 1.7, 1.8]  # seconds

# Data partition and sample extraction
train, test = preprocessing.partition_data_csv(
    terr_dfs,
    summary,
    PART_WINDOW,
    N_FOLDS,
    random_state=21,
)


# Data augmentation parameters
# 0 < STRIDE < MOVING_WINDOWS
STRIDE = 0.1  # seconds
# If True, balance the classes while augmenting
# If False, imbalance the classes while augmenting
HOMOGENEOUS_AUGMENTATION = True

# CNN parameters
cnn_par = {
    "time_window": 0.4,
    "time_ovrlap": 0.2,
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
svm_par = {"n_stat_mom": 4}

svm_train_opt = {
    "kernel_function": "poly",
    "poly_degree": 4,
    "kernel_scale": "auto",
    "box_constraint": 100,
    "standardize": True,
    "coding": "onevsone",
}

# Save results name
save_name = "TrainingResults_1"

# Model settings
# models = ["CNN", "LSTM", "CLSTM", "SVM"]
base_models = ["SVM"]

for MW in MOVING_WINDOWS:
    aug_train, aug_test = preprocessing.augment_data(
        train,
        test,
        summary,
        moving_window=MW,
        stride=STRIDE,
        homogeneous=HOMOGENEOUS_AUGMENTATION,
    )

    print(f"Training models for a sampling window of {MW} seconds")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    results = {}
    for model in base_models:
        if model == "CNN":
            train_mcs, test_mcs = preprocessing.apply_multichannel_spectogram(
                aug_train,
                aug_test,
                summary,
                cnn_par["time_window"],
                cnn_par["time_overlap"],
            )
            # results[model] = {
            #     f"SampWindow_{samp_window * 1000}ms": Conv_NeuralNet(
            #         train_mcs, test_mcs, cnn_par, cnn_train_opt
            #     )
            # }
        #     elif model == "LSTM":
        #         train_ds, test_ds = DownSample_Data(aug_train, aug_test, channels)
        #         results[model] = {
        #             f"SampWindow_{samp_window * 1000}ms": LSTM_RecurrentNet(
        #                 train_ds, test_ds, lstm_par, lstm_train_opt
        #             )
        #         }
        #     elif model == "CLSTM":
        #         train_ds, test_ds = DownSample_Data(aug_train, aug_test, channels)
        #         results[model] = {
        #             f"SampWindow_{samp_window * 1000}ms": CLSTM_RecurrentNet(
        #                 train_ds, test_ds, clstm_par, clstm_train_opt
        #             )
        #         }
        elif model == "SVM":
            results[model] = {
                f"SampWindow_{MW * 1000}ms": models.support_vector_machine(
                    aug_train,
                    aug_test,
                    summary,
                    svm_par["n_stat_mom"],
                    svm_train_opt,
                    random_state=21,
                )
            }

    # Store channels settings
    # results["Channels"] = channels

    # # Store terrain labels
    # terrains = [f.stem for f in data_dir.iterdir() if f.is_dir()]
    # results["TerLabls"] = terrains

    # np.save(
    #     results / f"{save_name}_{int(samp_window * 1000)}ms.npy",
    #     results,
    # )
