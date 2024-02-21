from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import signal as sig

from utils.constants import ch_cols

if TYPE_CHECKING:
    from typing import Tuple

    ExperimentData = dict[str, pd.DataFrame | np.ndarray]
    FrequencyData = dict[str, np.ndarray]


def multichannel_spectrogram(
    signal_cell: ExperimentData,
    summary: pd.DataFrame,
    mw: float,
    tw: float,
    to: float,
    hamming: bool = False,
) -> FrequencyData:
    # Output dictionary
    sens_mcs = {sens: {} for sens in signal_cell.keys()}

    # Input_cell is time
    for sens, sens_data in signal_cell.items():
        sf = summary["sampling_freq"].loc[sens]
        mcs, _, tgrid, fgrid = spectrogram(sens_data, sf, mw, tw, to, hamming)
        sens_mcs[sens]["fgrid"] = fgrid
        sens_mcs[sens]["tgrid"] = tgrid
        sens_mcs[sens]["spect"] = mcs

    # Highest sampling frequency
    hf_sensor = summary["sampling_freq"].idxmax()
    hf = summary["sampling_freq"].max()
    # Other sensors are low frequency
    lf_sensors = tuple(sens for sens in summary.index.values if sens != hf_sensor)

    pad_mcs = {hf_sensor: sens_mcs[hf_sensor]}
    hf_shape = pad_mcs[hf_sensor]["spect"].shape

    # Pad arrays
    for lf_sens in lf_sensors:
        if summary.loc[lf_sens].sampling_freq == hf:
            pad_mcs[lf_sens] = sens_mcs[lf_sens]
            continue
        lf_data = sens_mcs[lf_sens]
        pad_mcs.setdefault(lf_sens, {})
        fgrid = lf_data["fgrid"]
        tgrid = lf_data["tgrid"]
        spect = lf_data["spect"]

        pad_mcs[lf_sens]["fgrid"] = pad_array(
            fgrid,
            hf_shape[1],
            axis=0,
        )
        pad_mcs[lf_sens]["tgrid"] = pad_array(
            tgrid,
            hf_shape[1],
            axis=0,
        )
        pad_mcs[lf_sens]["spect"] = np.stack(
            [
                pad_array(
                    lay,
                    hf_shape[1],
                    axis=0,
                )
                for lay in spect
            ],
            axis=0,
        )

    # Join all channels
    multichannel = np.concatenate(
        [pad_mcs[lf_sens]["spect"] for lf_sens in lf_sensors],
        axis=3,
    )
    multichannel = np.concatenate([pad_mcs[hf_sensor]["spect"], multichannel], axis=3)

    freqgrid = np.dstack(
        [pad_mcs[lf_sens]["fgrid"][:, :, None] for lf_sens in lf_sensors]
    )
    freqgrid = np.dstack([pad_mcs[hf_sensor]["fgrid"][:, :, None], freqgrid])
    timegrid = np.dstack(
        [pad_mcs[lf_sens]["tgrid"][:, :, None] for lf_sens in lf_sensors]
    )
    timegrid = np.dstack([pad_mcs[hf_sensor]["tgrid"][:, :, None], timegrid])

    labels = signal_cell[hf_sensor][:, 0, ch_cols["terr_idx"]]

    return {"data": multichannel, "freq": freqgrid, "time": timegrid, "label": labels}


def spectrogram(
    data: np.ndarray,
    sampling_freq: float,
    moving_window: float,
    tw: float,
    to: float,
    hamming: bool = False,
) -> Tuple[np.array]:
    time = data[:, :, ch_cols["time"]]
    twto = tw - to
    n_windows = (moving_window - tw) // (twto) + 1
    time_part = time[0, :]
    t0 = time_part[0] + twto * np.arange(n_windows)
    t1 = t0 + tw
    win_len = int(tw * sampling_freq)

    # TODO: Use ShortTimeFFT for better results
    win_boxcar = sig.windows.boxcar(win_len, sym=False)
    win_hamming = sig.windows.hamming(win_len, sym=False)
    hop = int(twto * sampling_freq)
    sft = sig.ShortTimeFFT.from_window(
        ("boxcar"),
        fs=sampling_freq,
        nperseg=int(tw * sampling_freq),
        noverlap=int(to * sampling_freq),
        scale_to="magnitude",
    )
    # print(sft.stft(data[:, :, 5:], axis=1).shape, n_windows)

    lim0 = np.abs(time_part - t0[:, None]).argmin(axis=1)
    # lim1 = np.abs(time_part - t1[:, None]).argmin(axis=1)
    limits = np.vstack([lim0, lim0 + win_len]).T
    overwin = limits[:, 1] - time_part.size
    limits[overwin > 0, 0] -= overwin[overwin > 0]
    limits[overwin > 0, 1] -= overwin[overwin > 0]

    # Remove this line before running CNNs
    # data = np.concatenate([data, np.zeros(data.shape[:2])[:, :, None]], axis=2)
    windows = [data[:, slice(*lim), 5:] for lim in limits]

    norms, phases = [], []
    for win in windows:
        mag, phase, freq = DFT(win, sampling_freq, hamming)
        norms.append(mag)
        phases.append(phase)

    time_grid, freq_grid = np.meshgrid(twto * np.arange(n_windows) + tw, freq)

    # 4D array : instances x frequencies x windows x channels
    mags = np.stack(norms, axis=2)
    angs = np.stack(phases)

    return mags, angs, time_grid, freq_grid


def DFT(
    signal: np.array,
    sampling_freq: float,
    hamming: bool = False,
) -> Tuple[np.ndarray]:
    """Single Sided Discrete Fourier Transform of a signal

    Args:
        signal (np.array): Signals array
        sampling_freq (float): Sampling frequency

    Returns:
        Tuple[np.ndarray]: DFT of the signals : Mag, Phase, Frequency
    """
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    fftsig = signal.copy()

    sigsize = fftsig.shape[1]

    if hamming:
        # Use a hamming window
        hamm = np.hamming(sigsize)[None, :, None]
        fftsig *= hamm

    dsft = np.fft.fft(fftsig, axis=1)
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


def pad_array(arr: np.ndarray, out_dim: int, axis: int) -> np.ndarray:
    """Pad array with a given output shape

    Args:
        arr (np.ndarray): Arrau
        out_dim (int): Output shape along axis
        axis (int): Pad array along axis

    Raises:
        ValueError: Output shape doesn't work for padding

    Returns:
        np.ndarray: Padded array
    """
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]

    old_shp = arr.shape
    out_shp = list(old_shp)
    out_shp[axis] = out_dim
    out_shape = tuple(out_shp)

    pad = np.zeros(shape=out_shape)
    assert out_dim > old_shp[axis], "padded size must be greater than older size"

    n_repeats = round(out_dim / old_shp[axis])
    pad_dim = n_repeats * old_shp[axis]

    if axis == 1:
        # Need to concatenate horizontally
        pad[:, :pad_dim] = np.repeat(arr, n_repeats, axis=axis)[:, :out_dim]
        pad[:, pad_dim:] = arr[:, -1][:, None]

    elif axis == 0:
        # Need to concatenate vertically

        pad[:pad_dim] = np.repeat(arr, n_repeats, axis=axis)[:out_dim]
        pad[pad_dim:] = arr[-1][None]

    else:
        raise NotImplementedError(f"Axis {axis} is not implemented")

    return pad
