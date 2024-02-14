from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if TYPE_CHECKING:
    from typing import Dict, List, Literal, Tuple

    ExperimentData = dict[str, pd.DataFrame | np.ndarray]


def get_mixed_recordings(
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
    # For all csv paths
    for csvpath in csv_paths:
        csv_type, run_idx = csvpath.stem.split("_")
        df = pd.read_csv(csvpath)

        # Filter channels based on 'channels'
        filt_cols = [k for k, v in summary["columns"][csv_type].items() if v]
        terr_df = df[["time", *filt_cols, "terrain"]].copy()

        # Add info as DataFrame columns
        terr_df["run_idx"] = int(run_idx)

        dfs.setdefault(csv_type, []).append(terr_df)

    sensor_dfs = {
        sens: pd.concat(sensor_df, ignore_index=True) for sens, sensor_df in dfs.items()
    }

    return sensor_dfs
