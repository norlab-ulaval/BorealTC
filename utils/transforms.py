from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from typing import Literal

    ExperimentData = dict[str, pd.DataFrame | np.ndarray]


def merge_dfs(
    data: ExperimentData,
    hf_sensor: str,
    mode: Literal["interpolation", "last"] = "interpolation",
) -> pd.DataFrame:
    # Unmerged columns
    unmerged = ["terrain", "run_idx"]

    # Highest sampling frequency
    hf_data = data[hf_sensor]
    # Other sensors are low frequency
    lf_sensors = tuple(sens for sens in data.keys() if sens != hf_sensor)
    common = {sens: sdata.copy() for sens, sdata in data.items()}

    merged = data[hf_sensor].copy()
    if mode == "interpolation":
        interp_dfs = {}
        for lf_sens in lf_sensors:
            lf_data = common[lf_sens].rename(columns={"time": f"{lf_sens}_time"})
            cols = [c for c in lf_data.columns if c not in unmerged]

            # Time column
            t = lf_data[f"{lf_sens}_time"]
            # Interpolation functions
            fs = {c: CubicSpline(t, lf_data[c]) for c in cols}

            # Interpolated df
            int_df = merged[["time"]].copy()

            # For each function
            for c, func in fs.items():
                int_df[c] = func(int_df.time)

            interp_dfs[lf_sens] = int_df.copy()

        for int_df in interp_dfs.values():
            merged = pd.merge(merged, int_df, on="time", how="left")

        merged = merged[[c for c in merged.columns if not c.endswith("_time")]]
    elif mode == "last":
        for lf_sens in lf_sensors:
            lf_data = common[lf_sens]
            cols = [c for c in lf_data.columns if c not in unmerged]

            # Merged data based on the last available value available for each lf sensor
            merged = merged.merge(lf_data[cols], how="outer", on="time")
            merged[cols] = merged[cols].ffill()
            merged = merged.dropna().reset_index(drop=True)
            merged["run_idx"] = merged.run_idx.astype(int)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented")

    return merged
