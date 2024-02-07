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
    # Input_cell is time
    for sens, sens_data in signal_cell.items():
        sf = summary["sampling_freq"].loc[sens]
        spectrogram(sens_data, sf, tw, to)
        print(sens, sf)

    # Highest sampling frequency
    hf_sensor = summary["sampling_freq"].idxmax()
    hf = summary["sampling_freq"].max()
    # Other sensors are low frequency
    lf_sensors = tuple(sens for sens in summary.index.values if sens != hf_sensor)

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
        n_repeats, diff = divmod(out_shape[1], old_shp[1])
        pad[:, : n_repeats * old_shp[1]] = np.tile(arr, (1, n_repeats))
        pad[:, n_repeats * old_shp[1] :] = arr[:, :diff]

    elif old_shp[1] == out_shape[1]:
        # Need to concatenate vertically
        assert out_shape[0] > old_shp[0], "padded size must be greater than older size"
        n_repeats, diff = divmod(out_shape[0], old_shp[0])
        pad[: n_repeats * old_shp[0], :] = np.tile(arr, (n_repeats, 1))
        pad[n_repeats * old_shp[0] :, :] = arr[:diff, :]

    else:
        raise ValueError("Pad one dimension at the time")

    return pad
