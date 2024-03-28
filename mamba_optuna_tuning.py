import numpy as np
import optuna
import os
import pandas as pd

from optuna.integration import PyTorchLightningPruningCallback
from pathlib import Path
from utils import models, preprocessing
from utils.preprocessing import downsample_terr_dfs

cwd = Path.cwd()

DATASET = os.environ.get("DATASET", "vulpi")  # 'husky' or 'vulpi' or 'combined'
COMBINED_PRED = os.environ.get("COMBINED_PRED_TYPE", "class")  # 'class' or 'dataset'

if DATASET == "husky":
    csv_dir = cwd / "data" / "borealtc"
elif DATASET == "vulpi":
    csv_dir = cwd / "data" / "vulpi"
elif DATASET == "combined":
    csv_dir = dict(vulpi=cwd / "data" / "vulpi", husky=cwd / "data" / "borealtc")

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

# Get recordings
if DATASET == "combined":
    summary = {
        key: pd.DataFrame({"columns": pd.Series(columns)}) for key in csv_dir.keys()
    }

    terr_dfs = {}
    terrains = []

    terr_df_husky = preprocessing.get_recordings(csv_dir["husky"], summary["husky"])
    terr_df_vulpi = preprocessing.get_recordings(csv_dir["vulpi"], summary["vulpi"])

    terr_df_husky, terr_df_vulpi = downsample_terr_dfs(
        terr_df_husky, summary["husky"], terr_df_vulpi, summary["vulpi"]
    )

    terr_dfs["husky"] = terr_df_husky
    terr_dfs["vulpi"] = terr_df_vulpi

    if COMBINED_PRED == "class":
        for key in csv_dir.keys():
            terrains += sorted(terr_dfs[key]["imu"].terrain.unique())
    elif COMBINED_PRED == "dataset":
        terrains = list(csv_dir.keys())
else:
    summary = pd.DataFrame({"columns": pd.Series(columns)})

    terr_dfs = preprocessing.get_recordings(csv_dir, summary)
    terrains = sorted(terr_dfs["imu"].terrain.unique())

# Set data partition parameters
NUM_CLASSES = len(terrains)
N_FOLDS = 5
PART_WINDOW = 5  # seconds
MOVING_WINDOW = 1.7

# merged = preprocessing.merge_upsample(terr_dfs, summary, mode="last")

STRIDE = 0.1  # seconds
HOMOGENEOUS_AUGMENTATION = True

if DATASET == "combined":
    train_folds = {}
    test_folds = {}

    # Data partition and sample extraction
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

    # Data augmentation
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

    # Data cleanup and normalization
    for k in range(N_FOLDS):
        aug_train_fold = {}
        aug_test_fold = {}

        for key in csv_dir.keys():
            _aug_train_fold, _aug_test_fold = preprocessing.cleanup_data(
                aug_train_folds[key][k], aug_test_folds[key][k]
            )
            _aug_train_fold, _aug_test_fold = preprocessing.normalize_data(
                _aug_train_fold, _aug_test_fold
            )

            aug_train_fold[key] = _aug_train_fold
            aug_test_fold[key] = _aug_test_fold

        # Adapt class labels for combination
        if COMBINED_PRED == "class":
            num_classes_vulpi = len(np.unique(aug_train_fold["vulpi"]["labels"]))
            aug_train_fold["husky"]["labels"] += num_classes_vulpi
            aug_test_fold["husky"]["labels"] += num_classes_vulpi
        elif COMBINED_PRED == "dataset":
            aug_train_fold["vulpi"]["labels"] = np.full_like(
                aug_train_fold["vulpi"]["labels"], 0
            )
            aug_test_fold["vulpi"]["labels"] = np.full_like(
                aug_test_fold["vulpi"]["labels"], 0
            )
            aug_train_fold["husky"]["labels"] = np.full_like(
                aug_train_fold["husky"]["labels"], 1
            )
            aug_test_fold["husky"]["labels"] = np.full_like(
                aug_test_fold["husky"]["labels"], 1
            )

        aug_train_folds[k] = aug_train_fold
        aug_test_folds[k] = aug_test_fold
else:
    # Data partition and sample extraction
    train_folds, test_folds = preprocessing.partition_data(
        terr_dfs,
        summary,
        PART_WINDOW,
        N_FOLDS,
        random_state=RANDOM_STATE,
    )

    # Data augmentation
    aug_train_folds, aug_test_folds = preprocessing.augment_data(
        train_folds,
        test_folds,
        summary,
        moving_window=MOVING_WINDOW,
        stride=STRIDE,
        homogeneous=HOMOGENEOUS_AUGMENTATION,
    )

    # Data cleanup and normalization
    for k in range(N_FOLDS):
        aug_train_fold, aug_test_fold = preprocessing.cleanup_data(
            aug_train_folds[k], aug_test_folds[k]
        )
        aug_train_fold, aug_test_fold = preprocessing.normalize_data(
            aug_train_fold, aug_test_fold
        )
        aug_train_folds[k] = aug_train_fold
        aug_test_folds[k] = aug_test_fold


def objective_mamba(trial: optuna.Trial):
    ssm_cfg_imu = {
        "d_state": trial.suggest_int("d_state_imu", 8, 64, step=8),
        "d_conv": trial.suggest_int("d_conv_imu", 2, 4),
        "expand": trial.suggest_int("expand_imu", 2, 16),
    }
    ssm_cfg_pro = {
        "d_state": trial.suggest_int("d_state_pro", 8, 64, step=8),
        "d_conv": trial.suggest_int("d_conv_pro", 2, 4),
        "expand": trial.suggest_int("expand_pro", 2, 16, step=2),
    }

    mamba_train_opt = {
        "d_model_imu": trial.suggest_int("d_model_imu", 8, 64, step=8),
        "d_model_pro": trial.suggest_int("d_model_pro", 8, 64, step=8),
        "norm_epsilon": trial.suggest_float("norm_epsilon", 1e-8, 1e-1, log=True),
        "valid_perc": 0.1,
        "init_learn_rate": trial.suggest_float("init_learn_rate", 1e-5, 1e-1, log=True),
        "learn_drop_factor": trial.suggest_float("learn_drop_factor", 0.1, 0.5),
        "reduce_lr_patience": trial.suggest_int("reduce_lr_patience", 2, 8, step=2),
        "max_epochs": trial.suggest_int("max_epochs", 10, 60, step=10),
        "minibatch_size": trial.suggest_int("minibatch_size", 16, 128, step=16),
        "valid_patience": trial.suggest_int("valid_patience", 4, 16, step=4),
        "valid_frequency": None,
        "gradient_threshold": trial.suggest_categorical(
            "gradient_threshold", [0, 0.1, 1, 2, 6, 10, None]
        ),
        "focal_loss": True,
        "focal_loss_alpha": trial.suggest_float("focal_loss_alpha", 0.0, 1.0),
        "focal_loss_gamma": trial.suggest_float("focal_loss_gamma", 0.0, 5.0),
        "num_classes": NUM_CLASSES,
        "out_method": "last_state",  # trial.suggest_categorical("out_method", ["max_pool", "last_state"])
    }

    results = {}
    results_per_fold = []

    for k in range(N_FOLDS):
        out = models.mamba_network(
            aug_train_folds[k],
            aug_test_folds[k],
            mamba_train_opt,
            ssm_cfg_imu,
            ssm_cfg_pro,
            dict(mw=MOVING_WINDOW, fold=k + 1, dataset=DATASET),
            # custom_callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
            random_state=RANDOM_STATE,
            test=False,
            logging=False,
        )
        results_per_fold.append(out)

    results["pred"] = np.hstack([r["pred"] for r in results_per_fold])
    results["true"] = np.hstack([r["true"] for r in results_per_fold])

    # val_acc**4 to increase importance vs num_params
    val_acc = (results["pred"] == results["true"]).mean().item() ** 4
    num_params = out["num_params"]

    return val_acc, num_params


model = "Mamba"
IMP_ANALYSIS = os.environ.get("IMP_ANALYSIS", False)
study_name = f"{model}_{DATASET}"
optuna_path = Path(f"results/{DATASET}/optuna")
optuna_path.mkdir(parents=True, exist_ok=True)
storage_name = f"sqlite:///{optuna_path}/{study_name}.db"
print(f"Using database {storage_name}")

OBJECTIVE = objective_mamba

if IMP_ANALYSIS:
    pruner = optuna.pruners.MedianPruner()
    sampler = optuna.samplers.RandomSampler(seed=420)

    study = optuna.create_study(
        directions=["maximize", "minimize"],
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
        directions=["maximize", "minimize"],
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        pruner=pruner,
    )
    study.optimize(OBJECTIVE, n_trials=None, catch=(RuntimeError,), n_jobs=4)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
