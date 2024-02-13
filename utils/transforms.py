from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from utils.constants import HuskyConstants

if TYPE_CHECKING:
    from typing import Literal, Tuple

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


def motion_power(df: pd.DataFrame) -> pd.Series:
    """Compute motion power

    Args:
        df (pd.DataFrame): Proprioceptive data

    Returns:
        pd.Series: Motion power
    """
    consts = HuskyConstants

    I_L, I_R = df.curL, df.curR
    vL, vR = df.velL, df.velR

    # Angular velocities
    wL = vL / consts.ugv_wr
    wR = vR / consts.ugv_wr

    Tmot_L = consts.motor_Kt * I_L * np.sign(wL)
    Tmot_R = consts.motor_Kt * I_R * np.sign(wR)

    TL = consts.gear_eta * consts.gear_ratio * Tmot_L
    TR = consts.gear_eta * consts.gear_ratio * Tmot_R

    PM_L = TL * wL
    PM_R = TR * wR

    P_motion = PM_L + PM_R
    return P_motion


def ssmr_power_model(df: pd.DataFrame) -> Tuple[pd.Series]:
    vL, vR = df.velL, df.velR

    Husky = HuskyConstants

    # Normal force
    p = Husky.ugv_mass * Husky.g / 4
    # 4 distances
    sumdist = np.sqrt(Husky.ugv_wl**2 + (Husky.ugv_wb**2 - Husky.ugv_Bs) ** 2)

    HS = 2 * df.wz.abs() * p * sumdist
    HR = vL.abs() + vR.abs()

    return HS, HR
