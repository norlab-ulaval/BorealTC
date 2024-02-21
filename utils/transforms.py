from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from utils import frequency
from utils.constants import HuskyConstants, ch_cols

if TYPE_CHECKING:
    from typing import Literal, Sequence, Tuple

    ExperimentData = dict[str, pd.DataFrame | np.ndarray]
    FrequencyData = dict[str, np.ndarray]


def merge_dfs(
    data: ExperimentData,
    hf_sensor: str,
    mode: Literal["interpolation", "last"] = "interpolation",
) -> pd.DataFrame:
    """Merge dataframes, based on high frequency

    Args:
        data (ExperimentData): Dictionary of experiment data
        hf_sensor (str): High frequency sensor
        mode (Literal['interpolation', 'last'], optional): Merge mode. Defaults to "interpolation".

    Raises:
        NotImplementedError: Merge mode is not implemented

    Returns:
        pd.DataFrame: Merged DataFrame
    """
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
    """Apply SSMR power model

    Args:
        df (pd.DataFrame): Sensor values dataframe

    Returns:
        Tuple[pd.Series]: HS and HR columns
    """
    vL, vR = df.velL, df.velR

    Husky = HuskyConstants

    # Normal force
    p = Husky.ugv_mass * Husky.g / 4
    # 4 distances
    sumdist = np.sqrt(Husky.ugv_wl**2 + (Husky.ugv_wb**2 - Husky.ugv_Bs) ** 2)

    HS = 2 * df.wz.abs() * p * sumdist
    HR = vL.abs() + vR.abs()

    return HS, HR


def transform_augment(
    data: ExperimentData,
    hf_sensor: str,
    terrains: Sequence[str],
    lf_sensors: Sequence[str],
    summary: pd.DataFrame,
    aug_windows: np.ndarray,
    moving_window: float,
) -> ExperimentData:
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


def transform_mcs(
    data: ExperimentData,
    summary: pd.DataFrame,
    moving_window: float,
    time_window: float,
    time_overlap: float,
    hamming: bool = False,
) -> FrequencyData:
    """Apply multichannel spectrogram on data

    Args:
        data (ExperimentData): Experiment data
        summary (pd.DataFrame): Summary DataFrame
        moving_window (float): Moving Window, in seconds
        time_window (float): Fourier Window duration, in seconds
        time_overlap (float): Overlap duration, in seconds
        hamming (bool, optional): Use a hamming window. Defaults to False.

    Returns:
        FrequencyData: Multi Channel Spectrogram
    """
    return frequency.multichannel_spectrogram(
        data,
        summary,
        moving_window,
        time_window,
        time_overlap,
        hamming,
    )
