import os
import pathlib
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.integration import PyTorchLightningPruningCallback

from utils import models, preprocessing

cwd = Path.cwd()

DATASET = os.environ.get("DATASET", "vulpi")  # 'husky' or 'vulpi' or 'combined'
COMBINED_PRED = os.environ.get("COMBINED_PRED_TYPE", "class")  # 'class' or 'dataset'

if DATASET == "husky":
    csv_dir = cwd / "norlab-data"
elif DATASET == "vulpi":
    csv_dir = cwd / "data"
elif DATASET == "combined":
    csv_dir = dict(
        vulpi=cwd / "data",
        husky=cwd / "norlab-data"
    )

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

    for key in csv_dir.keys():
        terr_dfs[key] = preprocessing.get_recordings(csv_dir[key], summary[key])

    if COMBINED_PRED == "class":
        for key in csv_dir.keys():
            terrains += sorted(terr_dfs[key]["imu"].terrain.unique())
    elif COMBINED_PRED == "dataset":
        terrains = list(csv_dir.keys())
else:
    terr_dfs = preprocessing.get_recordings(csv_dir, summary)
    terrains = sorted(terr_dfs["imu"].terrain.unique())

# Set data partition parameters
NUM_CLASSES = len(terrains)
N_FOLDS = 5
PART_WINDOW = 5  # seconds
MOVING_WINDOW = 1.5

# merged = preprocessing.merge_upsample(terr_dfs, summary, mode="last")

STRIDE = 0.1  # seconds
HOMOGENEOUS_AUGMENTATION = True


def objective_cnn(trial: optuna.Trial):
    cnn_par = {
        "num_classes": NUM_CLASSES,
        "time_window": 0.4,
        "time_overlap": 0.2,
        "filter_size": trial.suggest_categorical("filter_size", [3, 5, 7]),
        "num_filters": trial.suggest_int("num_filters", 3, 32),
        "hamming": trial.suggest_categorical("hamming", [True, False]),
    }

    cnn_train_opt = {
        "valid_perc": 0.1,
        "init_learn_rate": trial.suggest_float("init_learn_rate", 1e-5, 1e-1, log=True),
        "learn_drop_factor": trial.suggest_float("learn_drop_factor", 0.1, 1.0),
        "max_epochs": 10,
        "minibatch_size": trial.suggest_int("minibatch_size", 5, 64),
        "valid_patience": trial.suggest_int("valid_patience", 5, 15),
        "reduce_lr_patience": trial.suggest_int("reduce_lr_patience", 2, 10),
        "valid_frequency": None,
        "gradient_threshold": trial.suggest_categorical(
            "gradient_threshold", [0, 0.1, 1, 2, 6, 10, None]
        ),
        "focal_loss": trial.suggest_categorical("focal_loss", [True, False]),
        "focal_loss_alpha": trial.suggest_float("focal_loss_alpha", 0.0, 1.0),
        "focal_loss_gamma": trial.suggest_float("focal_loss_gamma", 0.0, 5.0),
    }

    (
        train_mcs_folds,
        test_mcs_folds,
    ) = preprocessing.apply_multichannel_spectogram(
        aug_train_folds,
        aug_test_folds,
        summary,
        MOVING_WINDOW,
        cnn_par["time_window"],
        cnn_par["time_overlap"],
        hamming=cnn_par["hamming"],
    )
    k = 1
    train_mcs, test_mcs = train_mcs_folds[k], test_mcs_folds[k]
    out = models.convolutional_neural_network(
        train_mcs,
        test_mcs,
        cnn_par,
        cnn_train_opt,
        dict(mw=MOVING_WINDOW, fold=k + 1, dataset=DATASET),
        custom_callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        random_state=RANDOM_STATE,
        test=False
    )
    return (out["pred"] == out["true"]).mean().item()


def objective_mamba(trial: optuna.Trial):
    mamba_par = {
        "d_model_imu": trial.suggest_int("d_model_imu", 8, 64, step=8),
        "d_model_pro": trial.suggest_int("d_model_pro", 8, 64, step=8),
        "norm_epsilon": trial.suggest_float("norm_epsilon", 1e-8, 1e-1, log=True)
    }

    ssm_cfg = {
        "d_state": trial.suggest_int("d_state", 8, 64, step=8),
        "d_conv": trial.suggest_int("d_conv", 2, 4),
        "expand": trial.suggest_int("expand", 2, 4),
    }

    mamba_train_opt = {
        "valid_perc": 0.1,
        "init_learn_rate": trial.suggest_float("init_learn_rate", 1e-5, 1e-1, log=True),
        "learn_drop_factor": trial.suggest_float("learn_drop_factor", 0.1, 0.5),
        "max_epochs": 30,
        "minibatch_size": trial.suggest_int("minibatch_size", 8, 64, step=8),
        "valid_patience": trial.suggest_int("valid_patience", 2, 10, step=2),
        "reduce_lr_patience": trial.suggest_int("reduce_lr_patience", 2, 10, step=2),
        "valid_frequency": None,
        "gradient_treshold": trial.suggest_categorical("gradient_threshold", [0, 0.1, 1, 2, 6, 10, None]),
        "focal_loss": True,
        "focal_loss_alpha": trial.suggest_float("focal_loss_alpha", 0.0, 1.0),
        "focal_loss_gamma": trial.suggest_float("focal_loss_gamma", 0.0, 5.0),
        "num_classes": NUM_CLASSES,
        "out_method": trial.suggest_categorical("out_method", ["max_pool", "last_state"])
    }

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
        )

    if DATASET == "combined":
        aug_train_folds = {}
        aug_test_folds = {}

        for key in csv_dir.keys():
            _aug_train_folds, _aug_test_folds = preprocessing.augment_data(
                train_folds[key],
                test_folds[key],
                summary[key],
                moving_window=MOVING_WINDOW,
                stride=STRIDE,
                homogeneous=HOMOGENEOUS_AUGMENTATION,
            )
            aug_train_folds[key] = _aug_train_folds
            aug_test_folds[key] = _aug_test_folds
    else:
        aug_train_folds, aug_test_folds = preprocessing.augment_data(
            train_folds,
            test_folds,
            summary,
            moving_window=MOVING_WINDOW,
            stride=STRIDE,
            homogeneous=HOMOGENEOUS_AUGMENTATION,
        )

    k = 0
    if DATASET == "combined":
        aug_train_fold = {}
        aug_test_fold = {}

        for key in csv_dir.keys():
            _aug_train_fold, _aug_test_fold = preprocessing.cleanup_data(aug_train_folds[key][k], aug_test_folds[key][k])
            _aug_train_fold, _aug_test_fold = preprocessing.normalize_data(_aug_train_fold, _aug_test_fold)

            aug_train_fold[key] = _aug_train_fold
            aug_test_fold[key] = _aug_test_fold
        
        if COMBINED_PRED == "class":
            num_classes_vulpi = len(np.unique(aug_train_fold["vulpi"]["labels"]))
            aug_train_fold["husky"]["labels"] += num_classes_vulpi
            aug_test_fold["husky"]["labels"] += num_classes_vulpi
        elif COMBINED_PRED == "dataset":
            aug_train_fold["vulpi"]["labels"] = np.full_like(aug_train_fold["vulpi"]["labels"], 0)
            aug_test_fold["vulpi"]["labels"] = np.full_like(aug_test_fold["vulpi"]["labels"], 0)
            aug_train_fold["husky"]["labels"] = np.full_like(aug_train_fold["husky"]["labels"], 1)
            aug_test_fold["husky"]["labels"] = np.full_like(aug_test_fold["husky"]["labels"], 1)
    else:
        aug_train_fold, aug_test_fold = preprocessing.cleanup_data(aug_train_folds[k], aug_test_folds[k])
        aug_train_fold, aug_test_fold = preprocessing.normalize_data(aug_train_fold, aug_test_fold)

    out = models.mamba_network(
        aug_train_fold,
        aug_test_fold,
        mamba_par,
        mamba_train_opt,
        ssm_cfg,
        dict(mw=MOVING_WINDOW, fold=k+1, dataset=DATASET),
        # custom_callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
        random_state=RANDOM_STATE,
        test=False
    )
    return (out["pred"] == out["true"]).mean().item()


def objective_lstm(trial: optuna.Trial):
    lstm_par = {
        "num_classes": NUM_CLASSES,
        "nHiddenUnits": trial.suggest_int("nHiddenUnits", 5, 50),
        "numLayers": trial.suggest_int("numLayers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
        "convolutional": trial.suggest_categorical("convolutional", [True, False]),
    }
    lstm_train_opt = {
        "valid_perc": 0.1,
        "init_learn_rate": trial.suggest_float("init_learn_rate", 1e-5, 1e-1, log=True),
        "learn_drop_factor": trial.suggest_float("learn_drop_factor", 0.1, 1.0),
        "max_epochs": 10,
        "minibatch_size": trial.suggest_int("minibatch_size", 5, 64),
        "valid_patience": trial.suggest_int("valid_patience", 5, 15),
        "reduce_lr_patience": trial.suggest_int("reduce_lr_patience", 2, 10),
        "valid_frequency": None,
        "gradient_threshold": trial.suggest_categorical(
            "gradient_threshold", [0, 0.1, 1, 2, 6, 10, None]
        ),
        "focal_loss": trial.suggest_categorical("focal_loss", [True, False]),
        "focal_loss_alpha": trial.suggest_float("focal_loss_alpha", 0.0, 1.0),
        "focal_loss_gamma": trial.suggest_float("focal_loss_gamma", 0.0, 5.0),
    }

    train_ds_folds, test_ds_folds = preprocessing.downsample_data(
        aug_train_folds,
        aug_test_folds,
        summary,
    )
    k = 1
    train_ds, test_ds = train_ds_folds[k], test_ds_folds[k]
    out = models.long_short_term_memory(
        train_ds,
        test_ds,
        lstm_par,
        lstm_train_opt,
        dict(mw=MOVING_WINDOW, fold=k + 1, dataset=DATASET),
        custom_callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
        test=False
    )
    return (out["pred"] == out["true"]).mean().item()


def objective_svm(trial: optuna.Trial):
    svm_par = {"n_stat_mom": 4}

    svm_train_opt = {
        "kernel_function": trial.suggest_categorical(
            "kernel_function", ["linear", "poly", "rbf", "sigmoid"]
        ),
        "poly_degree": trial.suggest_int("poly_degree", 2, 10),
        "kernel_scale": "auto",
        "box_constraint": trial.suggest_float("box_constraint", 1e-5, 1e5, log=True),
        "standardize": True,
        "coding": "onevsone",
    }
    results = models.support_vector_machine(
        aug_train_folds,
        aug_test_folds,
        summary,
        svm_par["n_stat_mom"],
        svm_train_opt,
        random_state=RANDOM_STATE,
    )
    acc = results["pred"] == results["true"]
    return acc.mean()


model = os.environ.get("MODEL", "CNN")  # 'SVM', 'CNN', 'LSTM', 'Mamba'
IMP_ANALYSIS = os.environ.get("IMP_ANALYSIS", False)
study_name = f"{model}_{DATASET}"
optuna_path = pathlib.Path(f"results/{DATASET}/optuna")
optuna_path.mkdir(parents=True, exist_ok=True)
storage_name = f"sqlite:///{optuna_path}/{study_name}.db"
print(f"Using database {storage_name}")

OBJECTIVE = None
if model == "CNN":
    OBJECTIVE = objective_cnn
elif model == "LSTM":
    OBJECTIVE = objective_lstm
elif model == "SVM":
    OBJECTIVE = objective_svm
elif model == "Mamba":
    OBJECTIVE = objective_mamba
else:
    raise ValueError(f"Model {model} not recognized")

if IMP_ANALYSIS:
    pruner = optuna.pruners.MedianPruner()
    sampler = optuna.samplers.RandomSampler(seed=420)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(OBJECTIVE, n_trials=20, catch=(RuntimeError,))

    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
else:
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        pruner=pruner,
    )
    study.optimize(OBJECTIVE, n_trials=500, catch=(RuntimeError,))

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
