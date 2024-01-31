from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if TYPE_CHECKING:
    ExperimentData = dict[str, pd.DataFrame]


def get_recordings_csv(
    data_dir: Path,
    summary: pd.DataFrame,
) -> ExperimentData:
    """Extract data from CSVs in data_dir and filter out the columns with `channels`

    ```
        data
        ├── CLASS1
        │   ├── imu_1.csv
        │   ├── imu_2.csv
        │   ├── ...
        │   ├── pro_1.csv
        │   ├── pro_2.csv
        │   └── ...
        └── CLASS2
            ├── imu_1.csv
            ├── imu_2.csv
            ├── ...
            ├── pro_1.csv
            ├── pro_2.csv
            └── ...
    ```

    Args:
        data_dir (Path): Path to the dataset. The direct childs of data_dir are terrain classes folders
        summary (pd.DataFrame): Summary dataframe

    Returns:
        ExperimentData: Dictionary of dataframes
            `{"imu": imu_dataframe, "pro": pro_dataframe}`
            Each dataframe has `terrain` and `exp_idx` columns.
    """
    # All terrains names
    terrains = [f.stem for f in data_dir.iterdir() if f.is_dir()]

    # CSV filepaths
    csv_paths = [*data_dir.rglob("*.csv")]

    sampling_freq = {}

    dfs = {}
    # For all csv paths
    for csvpath in csv_paths:
        terrain = csvpath.parent.stem
        csv_type, run_idx = csvpath.stem.split("_")
        df = pd.read_csv(csvpath)

        # Filter channels based on 'channels'
        filt_cols = [k for k, v in summary["columns"][csv_type].items() if v]
        terr_df = df[["time", *filt_cols]].copy()

        # Add info as DataFrame columns
        terr_df["terrain"] = terrain
        terr_df["run_idx"] = int(run_idx)

        freq = int(1 / terr_df.time.diff().min())
        sampling_freq.setdefault(csv_type, freq)

        dfs.setdefault(csv_type, []).append(terr_df)

    sensor_dfs = {
        sens: pd.concat(sensor_df, ignore_index=True) for sens, sensor_df in dfs.items()
    }

    summary["sampling_freq"] = pd.Series(sampling_freq)

    return sensor_dfs


def partition_data_csv(
    data: ExperimentData,
    summary: pd.DataFrame,
    partition_duration: float,
    n_splits: int = 5,
    random_state: int | None = None,
):
    # Highest sampling frequency
    hf_sensor = summary["sampling_freq"].idxmax()
    hf = summary["sampling_freq"].max()
    # Other sensors are low frequency
    lf_sensors = tuple(sens for sens in data.keys() if sens != hf_sensor)

    # Time (s) / window * Sampling freq = samples / window
    wind_length = int(partition_duration * hf)

    # Create partition windows
    partitions = {hf_sensor: {}, **{s: {} for s in lf_sensors}}

    # Data from the high frequency sensor
    hf_data = data[hf_sensor]
    terrains = hf_data.terrain.unique().tolist()

    for terr in terrains:
        hf_terr = hf_data[hf_data.terrain == terr]
        exp_idxs = sorted(hf_terr.run_idx.unique())
        for exp_idx in exp_idxs:
            hf_exp = hf_terr[hf_terr.run_idx == exp_idx].copy().reset_index(drop=True)

            # Get limits, avoid selecting incomplete partitions
            starts = np.arange(0, hf_exp.shape[0], wind_length)
            starts = starts[(starts + wind_length) < hf_exp.shape[0]]
            limits = np.vstack([starts, starts + wind_length]).T

            # Get multiple windows
            windows = [
                hf_exp.iloc[slice(*lim)].assign(win_idx=win_idx)
                for win_idx, lim in enumerate(limits)
            ]
            tlimits = [np.array([w.time.min(), w.time.max()]) for w in windows]
            partitions[hf_sensor].setdefault(terr, []).extend(windows)

            # Slice each lf sensor based on the time from the hf windows
            for sens in lf_sensors:
                lf_data = data[sens]
                lf_terr = lf_data[lf_data.terrain == terr]
                lf_exp = lf_terr[lf_terr.run_idx == exp_idx]
                lf_exp = lf_exp.copy().reset_index(drop=True)
                lf_time = lf_exp.time.to_numpy()[None, :]

                indices = np.array(
                    [np.abs(lf_time - tlim[:, None]).argmin(axis=1) for tlim in tlimits]
                )
                indices[:, 1] += 1
                lf_win = [
                    lf_exp.iloc[slice(*lim)].assign(win_idx=win_idx)
                    for win_idx, lim in enumerate(indices)
                ]
                partitions[sens].setdefault(terr, []).extend(lf_win)

    columns = partitions[hf_sensor][terrains[0]][0].columns.values
    unified = {
        sens: np.vstack([sens_data[terr] for terr in terrains])
        for sens, sens_data in partitions.items()
    }
    n_windows = unified[hf_sensor].shape[0]

    # Split with K folds
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    train_data, test_data = [], []

    for fold_idx, (fold_train_idx, fold_test_idx) in enumerate(
        skf.split(np.arange(n_windows), np.arange(n_windows))
    ):
        train_data.append(
            {
                sens: sens_data[fold_train_idx, :, :]
                for sens, sens_data in unified.items()
            }
        )
        test_data.append(
            {
                sens: sens_data[fold_test_idx, :, :]
                for sens, sens_data in unified.items()
            }
        )

    return train_data, test_data


def augment_data(train_dat, test_dat, summary, w, AUG):
    # Find the channel "c" providing data at higher frequency "sf" to be used
    # as a reference for windowing operation
    channel_names = summary.index.values
    print(channel_names)

    return

    c = 0
    sf = channels[channel_names[c]]["sf"]
    for i in range(1, len(channel_names)):
        if channels[channel_names[i]]["sf"] > sf:
            sf = channels[channel_names[i]]["sf"]
            c = i

    # Find the minimum sampling frequency channel
    FN = list(train_dat.keys())
    TotLabl = np.concatenate([train_dat[FN[0]]["labl"], test_dat[FN[0]]["labl"]])
    NumTer = int(np.max(TotLabl))
    SizTer = np.zeros(NumTer)

    for i in range(1, NumTer + 1):
        SizTer[i - 1] = np.sum(TotLabl == i)

    MinNum = int(np.min(SizTer))

    # Decide whether to augment the data homogeneously or not
    if AUG["same"] == 1:
        TotSmp = len(GenSmp) * MinNum
        TerSmp = np.floor(TotSmp / SizTer)
        TerSli = (1 / channels[channel_names[c]]["sf"]) * (
            (Samp.shape[0] - int(w * channels[channel_names[c]]["sf"])) / TerSmp
        )
    else:
        TerSli = np.full(NumTer, AUG["sliding_window"])

    # Augment the data using the appropriate sliding window for different
    # terrains or the same for every terrain depending on AUG.same
    AugTrain = {}
    AugTest = {}

    for i in range(Kfold):
        k = 0
        for j in range(len(train_dat[FN[i]]["data"])):
            sli = TerSli[int(train_dat[FN[i]]["labl"][j] - 1)]
            strt = 0
            stop = int(strt + w * channels[channel_names[c]]["sf"]) - 1
            while stop < train_dat[FN[i]]["data"][j][c].shape[0]:
                AugTrain[FN[i]]["data"][k][c] = train_dat[FN[i]]["data"][j][c][
                    strt:stop, :
                ]
                AugTrain[FN[i]]["time"][k][c] = train_dat[FN[i]]["time"][j][c][
                    strt:stop
                ]
                AugTrain[FN[i]]["labl"][k] = train_dat[FN[i]]["labl"][j]

                t0 = train_dat[FN[i]]["time"][j][c][strt]
                t1 = train_dat[FN[i]]["time"][j][c][stop]
                for s in range(len(channel_names)):
                    if s != c:
                        e0 = np.argmin(
                            np.abs(t0 - train_dat[FN[i]]["time"][j][channel_names[s]])
                        )
                        e1 = np.argmin(
                            np.abs(t1 - train_dat[FN[i]]["time"][j][channel_names[s]])
                        )
                        AugTrain[FN[i]]["data"][k][s] = train_dat[FN[i]]["data"][j][s][
                            e0:e1, :
                        ]
                        AugTrain[FN[i]]["time"][k][s] = train_dat[FN[i]]["time"][j][s][
                            e0:e1
                        ]
                        # Make the dimensions homogeneous
                        if AugTrain[FN[i]]["data"][k][s].shape[0] > int(
                            round(w * channels[channel_names[s]]["sf"])
                        ):
                            AugTrain[FN[i]]["data"][k][s] = np.delete(
                                AugTrain[FN[i]]["data"][k][s], -1, axis=0
                            )
                            AugTrain[FN[i]]["time"][k][s] = np.delete(
                                AugTrain[FN[i]]["time"][k][s], -1
                            )
                strt = int(strt + sli * channels[channel_names[c]]["sf"])
                stop = int(strt + w * channels[channel_names[c]]["sf"]) - 1
                k += 1

        k = 0
        for j in range(len(test_dat[FN[i]]["data"])):
            sli = TerSli[int(test_dat[FN[i]]["labl"][j] - 1)]
            strt = 0
            stop = int(strt + w * channels[channel_names[c]]["sf"]) - 1
            while stop < test_dat[FN[i]]["data"][j][c].shape[0]:
                AugTest[FN[i]]["data"][k][c] = test_dat[FN[i]]["data"][j][c][
                    strt:stop, :
                ]
                AugTest[FN[i]]["time"][k][c] = test_dat[FN[i]]["time"][j][c][strt:stop]
                AugTest[FN[i]]["labl"][k] = test_dat[FN[i]]["labl"][j]

                t0 = test_dat[FN[i]]["time"][j][c][strt]
                t1 = test_dat[FN[i]]["time"][j][c][stop]
                for s in range(len(channel_names)):
                    if s != c:
                        e0 = np.argmin(
                            np.abs(t0 - test_dat[FN[i]]["time"][j][channel_names[s]])
                        )
                        e1 = np.argmin(
                            np.abs(t1 - test_dat[FN[i]]["time"][j][channel_names[s]])
                        )
                        AugTest[FN[i]]["data"][k][s] = test_dat[FN[i]]["data"][j][s][
                            e0:e1, :
                        ]
                        AugTest[FN[i]]["time"][k][s] = test_dat[FN[i]]["time"][j][s][
                            e0:e1
                        ]
                        # Make the dimensions homogeneous
                        if AugTest[FN[i]]["data"][k][s].shape[0] > int(
                            round(w * channels[channel_names[s]]["sf"])
                        ):
                            AugTest[FN[i]]["data"][k][s] = np.delete(
                                AugTest[FN[i]]["data"][k][s], -1, axis=0
                            )
                            AugTest[FN[i]]["time"][k][s] = np.delete(
                                AugTest[FN[i]]["time"][k][s], -1
                            )
                strt = int(strt + sli * channels[channel_names[c]]["sf"])
                stop = int(strt + w * channels[channel_names[c]]["sf"]) - 1
                k += 1
