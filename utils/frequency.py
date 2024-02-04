from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
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
    pass
