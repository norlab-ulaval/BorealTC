from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import StratifiedKFold

from utils import preprocessing
from utils.constants import ch_cols

if TYPE_CHECKING:
    from typing import Dict, List, Literal, Tuple

    ExperimentData = dict[str, pd.DataFrame | np.ndarray]


def mx_get_recordings(
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
    # CSV filepaths
    csv_paths = [*data_dir.rglob("*.csv")]

    dfs = {}
    sampling_freq = {}
    # For all csv paths
    for csvpath in csv_paths:
        csv_type, run_idx = csvpath.stem.split("_")
        df = pd.read_csv(csvpath)

        # Filter channels based on 'channels'
        filt_cols = [k for k, v in summary["columns"][csv_type].items() if v]
        terr_df = df[["time", *filt_cols, "terrain"]].copy()

        # Add info as DataFrame columns
        terr_df["run_idx"] = int(run_idx)
        terr_df["terrain"] = terr_df.terrain.astype(str)

        freq = round(1 / terr_df.time.diff().min(), 1)
        sampling_freq.setdefault(csv_type, freq)

        dfs.setdefault(csv_type, []).append(terr_df)

    sensor_dfs = {
        sens: pd.concat(sensor_df, ignore_index=True) for sens, sensor_df in dfs.items()
    }

    summary["sampling_freq"] = pd.Series(sampling_freq)

    return sensor_dfs


def mx_partition_data(
    data: ExperimentData,
    summary: pd.DataFrame,
    moving_window: float,
    n_splits: int | None = 5,
    random_state: int | None = None,
) -> ExperimentData:
    partitioned = preprocessing.partition_data(
        data,
        summary,
        moving_window,
        n_splits=None,
        random_state=random_state,
    )

    # Highest sampling frequency
    lf_sensor = summary["sampling_freq"].idxmin()
    lf = summary["sampling_freq"].min()
    # Other sensors are low frequency
    hf_sensors = tuple(sens for sens in data.keys() if sens != lf_sensor)

    # Partitions
    partitions = {}

    # Size of lf windows
    lf_sz = partitioned[lf_sensor].shape[1]

    lf_data = data[lf_sensor]
    lf_data["terr_idx"] = -1
    lf_data["win_idx"] = -1
    lf_cols = lf_data.columns.tolist()
    lf_c = [*np.take(lf_cols, (-4, -2, -3, -1, 0)), *lf_cols[1:-4]]

    lf_arr = lf_data[lf_c].to_numpy()
    lf_idxs = sliding_window_view(lf_data.index, lf_sz)
    lf_wins = lf_arr[lf_idxs, :]

    lf_tlim = lf_wins[:, 0, ch_cols["time"]]

    partitions[lf_sensor] = lf_wins.copy()

    for hf_sens in hf_sensors:
        hf_sz = partitioned[hf_sens].shape[1]

        hf_data = data[hf_sens]
        hf_data["terr_idx"] = -1
        hf_data["win_idx"] = -1
        hf_cols = hf_data.columns.tolist()
        hf_c = [*np.take(hf_cols, (-4, -2, -3, -1, 0)), *hf_cols[1:-4]]

        hf_arr = hf_data[hf_c].to_numpy()
        hf_time = hf_arr[:, ch_cols["time"]]

        hf_tpad = np.repeat(hf_time[None, :], lf_tlim.shape, axis=0)
        lf_tpad = np.repeat(lf_tlim[:, None], hf_time.shape, axis=1)
        indices = lf_tpad[: lf_tlim // 2, :] - hf_tpad[: lf_tlim // 2, :]
        print(indices.shape)

        hf_idxs = sliding_window_view(hf_data.index, hf_sz)
        hf_wins = hf_arr[hf_idxs, :]

        partitions[hf_sens] = hf_wins.copy()

        pass

    pass
