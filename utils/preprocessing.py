from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils.transforms import merge_dfs

if TYPE_CHECKING:
    from typing import Dict, List, Literal, Tuple

    ExperimentData = dict[str, pd.DataFrame | np.ndarray]

from utils import frequency
from utils.constants import ch_cols


def get_recordings(
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

        freq = round(1 / terr_df.time.diff().min(), 1)
        sampling_freq.setdefault(csv_type, freq)

        dfs.setdefault(csv_type, []).append(terr_df)

    sensor_dfs = {
        sens: pd.concat(sensor_df, ignore_index=True) for sens, sensor_df in dfs.items()
    }
    sensor_dfs = {
        sens: sens_data[sens_data.terrain != "MIXED"]
        for sens, sens_data in sensor_dfs.items()
    }

    summary["sampling_freq"] = pd.Series(sampling_freq)

    return sensor_dfs


def partition_data(
    data: ExperimentData,
    summary: pd.DataFrame,
    partition_duration: float,
    n_splits: int = 5,
    random_state: int | None = None,
) -> Tuple[List[ExperimentData]]:
    """_summary_

    Args:
        data (ExperimentData): Dictionary of pandas DataFrames
        summary (pd.DataFrame): Summary DataFrame
        partition_duration (float): Duration of a partition window (in seconds).
        n_splits (int, optional): Number of folds. Defaults to 5.
        random_state (int | None, optional): Random state. Defaults to None.

    Returns:
        List[ExperimentData]: List of dicts with numpy arrays of dimensions (Number partitions x time x channels)
    """
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
            tlimits = np.array([w.time.min() for w in windows])
            partitions[hf_sensor].setdefault(terr, []).extend(windows)

            # Slice each lf sensor based on the time from the hf windows
            for sens in lf_sensors:
                lf_data = data[sens]
                lf_terr = lf_data[lf_data.terrain == terr].assign(terr_idx=terr_idx)
                lf_exp = lf_terr[lf_terr.run_idx == exp_idx]
                lf_exp = lf_exp.copy().reset_index(drop=True)
                lf_time = lf_exp.time.to_numpy()[None, :]

                sf = summary.sampling_freq.loc[sens]
                lf_winlen = int(partition_duration * sf)

                indices = np.abs(lf_time - tlimits[:, None]).argmin(axis=1)
                indices = np.vstack([indices, indices + lf_winlen]).T

                overwin = indices[:, 1] - len(lf_exp.index)
                indices[overwin > 0, 0] -= overwin[overwin > 0]
                indices[overwin > 0, 1] -= overwin[overwin > 0]
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
    unified = {}
    for sens, sens_data in partitions.items():
        # print(sens, [sens_data[terr][0].shape for terr in terrains])
        unified[sens] = np.vstack([sens_data[terr] for terr in terrains])
    # n_windows = unified[hf_sensor].shape[0]
    labels = unified[hf_sensor][:, 0, terr_col][:, 0, 0]
    # for sens, sens_data in unified.items():
    #     print(sens, sens_data.shape, (sens_data[:, 0, :][:, 0] == labels).all())

    # TODO: split elsewhere ?

    # Split data with K folds
    rng = np.random.RandomState(random_state)
    skf = StratifiedKFold(n_splits=n_splits, random_state=rng, shuffle=True)

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
) -> Tuple[List[ExperimentData]]:
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

    def data_augmentation(data: ExperimentData) -> ExperimentData:
        # Augment the data using the appropriate sliding window for different
        # terrains or the same for every terrain depending on homogeneous

        # For every terrain
        Kterr = {}

        hf_data = data[hf_sensor]

        for terr_idx, terr in enumerate(terrains):
            # Get partitions for label terrain
            terr_mask = hf_data[:, 0, ch_cols["terrain"]] == terr
            hf_terr = hf_data[terr_mask]

            # Sliding window for the terrain class
            sli_len = aug_windows[terr_idx]
            n_slides = n_slides_terr[terr]

            # Slice the array based on the slide length
            starts = sli_len * np.arange(n_slides)
            # starts = np.arange(0, PW_len - MW_len, sli_len)
            # starts = starts[(starts + MW_len) < PW_len]
            limits = np.vstack([starts, starts + MW_len]).T
            # TODO: Check number of strides per partition
            # if K_idx == 4:
            #     print(K_idx, terr_idx, n_slides, n_slides * hf_terr_train.shape[0])
            # print(strides_min, n_slides, limits.shape[0])

            # Get slices for each limit
            hf_sli = [hf_terr[:, slice(*lim), :] for lim in limits]

            # Get time values
            hf_tlim = hf_terr[0, limits, ch_cols["time"]]

            # Stack all slices together
            hf_sli = np.vstack(hf_sli)

            Kterr.setdefault(hf_sensor, []).append(hf_sli)

            for lf_sens in lf_sensors:
                lf_data = data[lf_sens]

                # Sampling frequency
                sf = summary.sampling_freq.loc[lf_sens]
                lf_win = int(moving_window * sf)

                # Select partitions based on terrain
                terr_mask = lf_data[:, 0, ch_cols["terrain"]] == terr
                lf_terr = lf_data[terr_mask]

                lf_time = lf_terr[0, :, ch_cols["time"]]
                indices = np.abs(lf_time - hf_tlim[:, [0]]).argmin(axis=1)
                lf_sli = [
                    lf_terr[:, lf_sli_idx : (lf_sli_idx + lf_win), :]
                    for lf_sli_idx in indices
                ]
                lf_sli = np.vstack(lf_sli)

                Kterr.setdefault(lf_sens, []).append(lf_sli)

        K_sli = {sens: np.vstack(sens_data) for sens, sens_data in Kterr.items()}

        return K_sli

    # For every fold
    for K_idx, (K_train, K_test) in enumerate(zip(train_dat, test_dat)):
        aug_train.append(data_augmentation(K_train))
        aug_test.append(data_augmentation(K_test))

    return aug_train, aug_test


def apply_multichannel_spectogram(
    train_dat: List[ExperimentData],
    test_dat: List[ExperimentData],
    summary: pd.DataFrame,
    time_window: float,
    time_overlap: float,
) -> Tuple[List[Dict[str, ExperimentData]]]:
    tw = time_window
    to = time_overlap

    mcs_train, mcs_test = [], []

    for K_idx, (K_train, K_test) in enumerate(zip(train_dat, test_dat)):
        mcs_train.append(frequency.multichannel_spectrogram(K_train, summary, tw, to))
        mcs_test.append(frequency.multichannel_spectrogram(K_test, summary, tw, to))

    return mcs_train, mcs_test


def downsample_data(
    train_dat: List[ExperimentData],
    test_dat: List[ExperimentData],
    summary: pd.DataFrame,
) -> Tuple[List[ExperimentData]]:
    # Highest sampling frequency
    lf_sensor = summary["sampling_freq"].idxmin()
    lf = summary["sampling_freq"].min()
    # Other sensors are low frequency
    hf_sensors = tuple(sens for sens in summary.index.values if sens != lf_sensor)

    def data_downsampling(data: ExperimentData) -> ExperimentData:
        # Augment the data using the appropriate sliding window for different
        # terrains or the same for every terrain depending on homogeneous
        # TODO take hf_sensors, and make it generic to more sensors
        imu = data["imu"]
        pro = data["pro"]
        imu_time = data["imu"][:, :, ch_cols["time"]]
        pro_time = data["pro"][:, :, ch_cols["time"]]

        new_data = {
            "imu": [],
            "pro": data["pro"],
        }
        for i in range(imu_time.shape[0]):
            idx = np.argmin(
                np.abs(pro_time[i, :, np.newaxis] - imu_time[i, np.newaxis]), axis=1
            )
            new_data["imu"].append(imu[i, idx])
        new_data["imu"] = np.array(new_data["imu"])
        return new_data

    train_ds, test_ds = [], []
    for K_idx, (K_train, K_test) in enumerate(zip(train_dat, test_dat)):
        train_ds.append(data_downsampling(K_train))
        test_ds.append(data_downsampling(K_test))

    return train_ds, test_ds


def merge_upsample(
    data: ExperimentData,
    summary: pd.DataFrame,
    mode: Literal["interpolation", "last"] = "interpolation",
) -> pd.DataFrame:
    """Upsample and merge low frequency sensors

    Args:
        data (ExperimentData): Experiment data
        summary (pd.DataFrame): Summary DataFrame
        mode (Literal["interpolation", "last"], optional): Merge mode. 'interpolation' for cubic spline interpolation, 'last' for interpolation based on the last available data for each hf time. Defaults to "interpolation".

    Returns:
        pd.DataFrame: Merged DataFrame
    """
    # Highest sampling frequency
    hf_sensor = summary["sampling_freq"].idxmax()
    hf = summary["sampling_freq"].max()
    # Other sensors are low frequency
    lf_sensors = tuple(sens for sens in data.keys() if sens != hf_sensor)

    merged = []

    hf_data = data[hf_sensor]
    terrains = sorted(hf_data.terrain.unique().tolist())
    for terr in terrains:
        terrdat = {sens: sdata[sdata.terrain == terr] for sens, sdata in data.items()}

        hf_terr = terrdat[hf_sensor]
        exp_idxs = sorted(hf_terr.run_idx.unique())
        for exp_idx in exp_idxs:
            exps = {
                sens: sdata[sdata.run_idx == exp_idx].copy().reset_index(drop=True)
                for sens, sdata in terrdat.items()
            }
            hf_exp = exps[hf_sensor]

            merged.append(merge_dfs(exps, hf_sensor, mode))

    return pd.concat(merged, ignore_index=True)
