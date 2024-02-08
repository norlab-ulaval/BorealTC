from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from utils.constants import ch_cols

if TYPE_CHECKING:
    from typing import Tuple

    ExperimentData = dict[str, pd.DataFrame | np.ndarray]


def multichannel_spectrogram(
    signal_cell: ExperimentData,
    summary: pd.DataFrame,
    tw: float,
    to: float,
):
    # Output dictionary
    sens_mcs = {sens: {} for sens in signal_cell.keys()}

    # Input_cell is time
    for sens, sens_data in signal_cell.items():
        sf = summary["sampling_freq"].loc[sens]
        mcs, _, tgrid, fgrid = spectrogram(sens_data, sf, tw, to)
        sens_mcs[sens]["fgrid"] = fgrid
        sens_mcs[sens]["tgrid"] = tgrid
        sens_mcs[sens]["spect"] = mcs
        print(sens, sf)

    # Highest sampling frequency
    hf_sensor = summary["sampling_freq"].idxmax()
    hf = summary["sampling_freq"].max()
    # Other sensors are low frequency
    lf_sensors = tuple(sens for sens in summary.index.values if sens != hf_sensor)

    pad_mcs = {hf_sensor: sens_mcs[hf_sensor]}
    hf_shape = pad_mcs[hf_sensor]["tgrid"].shape

    for lf_sens in lf_sensors:
        if summary.iloc[lf_sens].sampling_freq == hf:
            pad_mcs[lf_sens] = sens_mcs[lf_sens]
            continue
        lf_data = sens_mcs[lf_sens]
        pad_mcs.setdefault(lf_sens, {})
        fgrid = lf_data["fgrid"]
        tgrid = lf_data["tgrid"]
        spect = lf_data["spect"]

        pad_mcs[lf_sens]["fgrid"] = pad_array(
            fgrid,
            hf_shape,
        )
        pad_mcs[lf_sens]["tgrid"] = pad_array(
            tgrid,
            hf_shape,
        )
        pad_mcs[lf_sens]["spect"] = [pad_array(mcs, hf_shape) for mcs in spect]

    exit(0)

    return (0, 0, 0)


def spectrogram(
    data: np.ndarray,
    sampling_freq: float,
    tw: float,
    to: float,
):
    time = data[:, :, ch_cols["time"]]
    twto = tw - to
    n_windows = ((time.shape[1] / sampling_freq) - tw) // (twto) + 1
    time_part = time[0, :]
    t0 = time_part[0] + twto * np.arange(n_windows)
    t1 = t0 + tw

    lim0 = np.abs(time_part - t0[:, None]).argmin(axis=1)
    lim1 = np.abs(time_part - t1[:, None]).argmin(axis=1)
    limits = np.vstack([lim0, lim1 + 1]).T

    windows = [data[:, slice(*lim), 5:] for lim in limits]

    norms, phases = [], []
    for win in windows:
        mag, phase, freq = DFT(win, sampling_freq)
        norms.append(mag)
        phases.append(phase)

    time_grid, freq_grid = np.meshgrid(twto * np.arange(n_windows) + tw, freq)

    return norms, phases, time_grid, freq_grid


def DFT(signal: np.array, sampling_freq: float) -> Tuple[np.ndarray]:
    """Single Sided Discrete Fourier Transform of a signal

    Args:
        signal (np.array): Signals array
        sampling_freq (float): Sampling frequency

    Returns:
        Tuple[np.ndarray]: DFT of the signals : Mag, Phase, Frequency
    """
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    sigsize = signal.shape[1]

    dsft = np.fft.fft(signal, axis=1)
    mag = np.absolute(dsft)
    dsft[mag < 1e-6] = 0
    ang = np.unwrap(np.angle(dsft))

    ssft = dsft[:, : sigsize // 2 + 1] / sigsize
    if sigsize % 2 == 0:
        ssft[:, 1:-1] *= 2
    else:
        ssft[:, 1:] *= 2

    magn = np.absolute(ssft)
    phase = ang[:, : ssft.shape[1]]
    freq = np.linspace(0, sampling_freq / 2, ssft.shape[1])

    return magn, phase, freq


def pad_array(arr: np.ndarray, out_shape: Tuple[int]) -> np.ndarray:
    """Pad array with a given output shape

    Args:
        arr (np.ndarray): Arrau
        out_shape (Tuple[int]): Output shape

    Raises:
        ValueError: Output shape doesn't work for padding

    Returns:
        np.ndarray: Padded array
    """
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    old_shp = arr.shape
    pad = np.zeros(shape=out_shape)

    if old_shp[0] == out_shape[0]:
        # Need to concatenate horizontally
        assert out_shape[1] > old_shp[1], "padded size must be greater than older size"

        n_repeats = round(out_shape[1] / old_shp[1])
        pad_dim = n_repeats * old_shp[1]

        pad[:, :pad_dim] = np.repeat(arr, n_repeats, axis=1)[:, : out_shape[1]]
        pad[:, pad_dim:] = arr[:, -1][:, None]

    elif old_shp[1] == out_shape[1]:
        # Need to concatenate vertically
        assert out_shape[0] > old_shp[0], "padded size must be greater than older size"

        n_repeats = round(out_shape[0] / old_shp[0])
        pad_dim = n_repeats * old_shp[0]

        pad[:pad_dim, :] = np.repeat(arr, n_repeats, axis=0)[: out_shape[0], :]
        pad[pad_dim:, :] = arr[-1, :][None, :]

    else:
        raise ValueError("Pad one dimension at the time")

    return pad
