from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if TYPE_CHECKING:
    ExperimentData = dict[str, pd.DataFrame]

ch_cols = dict(zip(("terrain", "terr_idx", "run_idx", "win_idx", "time"), range(5)))
first_last = np.array([0, -1])


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
    wind_len = int(partition_duration * hf)

    # Create partition windows
    partitions = {hf_sensor: {}, **{s: {} for s in lf_sensors}}

    # Data from the high frequency sensor
    hf_data = data[hf_sensor]
    terrains = sorted(hf_data.terrain.unique().tolist())

    for terr_idx, terr in enumerate(terrains):
        hf_terr = hf_data[hf_data.terrain == terr].assign(terr_idx=terr_idx)
        exp_idxs = sorted(hf_terr.run_idx.unique())
        for exp_idx in exp_idxs:
            hf_exp = hf_terr[hf_terr.run_idx == exp_idx].copy().reset_index(drop=True)

            # Get limits, avoid selecting incomplete partitions
            starts = np.arange(0, hf_exp.shape[0], wind_len)
            starts = starts[(starts + wind_len) <= hf_exp.shape[0]]
            limits = np.vstack([starts, starts + wind_len]).T

            # Get multiple windows
            windows = [
                hf_exp.iloc[slice(*lim)].assign(win_idx=win_idx)
                for win_idx, lim in enumerate(limits)
            ]
            hf_cols = windows[0].columns.tolist()
            hf_c = [*np.take(hf_cols, (-4, -2, -3, -1, 0)), *hf_cols[1:-4]]
            windows = [w[hf_c] for w in windows]
            tlimits = [np.array([w.time.min(), w.time.max()]) for w in windows]
            partitions[hf_sensor].setdefault(terr, []).extend(windows)

            # Slice each lf sensor based on the time from the hf windows
            for sens in lf_sensors:
                lf_data = data[sens]
                lf_terr = lf_data[lf_data.terrain == terr].assign(terr_idx=terr_idx)
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
                lf_cols = lf_win[0].columns.tolist()
                lf_c = [*np.take(lf_cols, (-4, -2, -3, -1, 0)), *lf_cols[1:-4]]
                lf_win = [w[lf_c] for w in lf_win]
                partitions[sens].setdefault(terr, []).extend(lf_win)

    hf_columns = partitions[hf_sensor][terrains[0]][0].columns.values
    terr_col = np.where(hf_columns == "terrain")
    # Number partitions x time x channels
    # terrain, terr_idx, run_idx, win_idx, time, <sensor_channels>
    unified = {
        sens: np.vstack([sens_data[terr] for terr in terrains])
        for sens, sens_data in partitions.items()
    }
    n_windows = unified[hf_sensor].shape[0]
    labels = unified[hf_sensor][:, 0, terr_col][:, 0, 0]
    # for sens, sens_data in unified.items():
    #     print(sens, sens_data.shape, (sens_data[:, 0, :][:, 0] == labels).all())

    # Split with K folds
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    train_data, test_data = [], []

    for fold_train_idx, fold_test_idx in skf.split(np.zeros(len(labels)), labels):
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


def augment_data(
    train_dat,
    test_dat,
    summary,
    moving_window: float,
    stride: float,
    homogeneous: bool,
):
    # Find the channel "c" providing data at higher frequency "sf" to be used
    # as a reference for windowing operation

    # Highest sampling frequency
    hf_sensor = summary["sampling_freq"].idxmax()
    hf = summary["sampling_freq"].max()
    # Other sensors are low frequency
    lf_sensors = tuple(sens for sens in summary.index.values if sens != hf_sensor)

    # Time (s) / window * Sampling freq = samples / window
    MW_len = int(moving_window * hf)
    ST_len = int(stride * hf)
    PW_len = train_dat[0][hf_sensor].shape[1]

    # Number of folds
    num_folds = len(train_dat)
    all_labels = np.hstack(
        [
            train_dat[0][hf_sensor][:, 0, 0],
            test_dat[0][hf_sensor][:, 0, 0],
        ]
    )
    terrains = np.sort(np.unique(all_labels))
    num_terrains = terrains.shape[0]

    if homogeneous:
        # Get number of windows per terrain
        terr_counts = pd.Series(all_labels).value_counts(sort=False).sort_index()
        min_count = terr_counts.min()

        # How many samples are generated for one partition window
        n_strides_part = (PW_len - MW_len) // ST_len
        # Maximum amount of samples / slides for the class with less partitions
        # strides_min = n_strides/part * n_partitions(small class)
        strides_min = n_strides_part * min_count

        # Number of slides for each terrain so that all terrains have strides_min slides
        n_slides_terr = (strides_min / terr_counts).astype(int)

        # Length of slide for each terrain to respect the number of slides/terr
        aug_strides_all = (PW_len - MW_len) // (n_slides_terr)
        aug_windows = aug_strides_all.to_numpy()

    else:
        # Use ST for all terrains
        aug_windows = np.full_like(terrains, int(stride * hf))

    aug_train, aug_test = [], []

    # Augment the data using the appropriate sliding window for different
    # terrains or the same for every terrain depending on homogeneous

    # For every fold
    for K_train, K_test in zip(train_dat, test_dat):
        # For every terrain
        # Ktrain_copy = {sens: sensKtrain.copy() for sens, sensKtrain in K_train.items()}
        # Ktest_copy = {sens: sensKtest.copy() for sens, sensKtest in K_test.items()}
        # aug_train.append(Ktrain_copy)
        # aug_test.append(Ktest_copy)

        K_sli_train = {}
        K_sli_test = {}

        hf_train = K_train[hf_sensor]
        hf_test = K_test[hf_sensor]

        for terr_idx, terr in enumerate(terrains):
            train_terr_mask = hf_train[:, 0, ch_cols["terrain"]] == terr
            test_terr_mask = hf_test[:, 0, ch_cols["terrain"]] == terr
            hf_terr_train = hf_train[train_terr_mask]
            hf_terr_test = hf_test[test_terr_mask]

            # Sliding window for the terrain class
            sli_len = aug_windows[terr_idx]
            n_slides = n_slides_terr[terr]

            # Slice the array based on the slide length
            starts = np.arange(0, PW_len, sli_len)
            starts = starts[(starts + MW_len) <= PW_len]
            limits = np.vstack([starts, starts + MW_len]).T
            print(strides_min, n_slides, limits.shape[0])

            # Get slices for each limit
            hf_sli_train = [hf_terr_train[:, slice(*lim), :] for lim in limits]
            hf_sli_test = [hf_terr_test[:, slice(*lim), :] for lim in limits]
            hf_sli_train = np.vstack(hf_sli_train)
            hf_sli_test = np.vstack(hf_sli_test)

            K_sli_train.setdefault(hf_sensor, []).append(hf_sli_train)
            K_sli_test.setdefault(hf_sensor, []).append(hf_sli_test)

            # Get time values
            t_limits_train = hf_sli_train[:, (0, -1), ch_cols["time"]]
            t_limits_test = hf_sli_test[:, (0, -1), ch_cols["time"]]

            for lf_sens in lf_sensors:
                lf_train = K_train[lf_sens]
                lf_test = K_test[lf_sens]

                train_terr_mask = lf_train[:, 0, ch_cols["terrain"]] == terr
                test_terr_mask = lf_test[:, 0, ch_cols["terrain"]] == terr
                lf_terr_train = lf_train[train_terr_mask]
                lf_terr_test = lf_test[test_terr_mask]

                # Time length
                win_len = lf_train.shape[1]

        aug_train.append(K_sli_train)
        aug_test.append(K_sli_test)

        pass

    return aug_train, aug_test

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
