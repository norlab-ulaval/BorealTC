import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils import mixed

cwd = Path.cwd()
csv_dir = cwd / "data" / "borealtc" / "MIXED"

RANDOM_STATE = 21

# Define channels
columns = {
    "imu": {
        "wx": True,
        "wy": True,
        "wz": True,
        "ax": True,
        "ay": True,
        "az": True,
    },
    "pro": {
        "velL": True,
        "velR": True,
        "curL": True,
        "curR": True,
    },
}
summary = pd.DataFrame({"columns": pd.Series(columns)})

# Get recordings
terr_dfs = mixed.mx_get_recordings(csv_dir, summary)

# Set data partition parameters
NUM_CLASSES = len(np.unique(terr_dfs["imu"].terrain))
N_FOLDS = 5
MOVING_WINDOWS = [1.5, 1.6, 1.7, 1.8]  # seconds

# Model settings
BASE_MODELS = ["SVM", "CNN", "LSTM", "CLSTM"]

print(f"Infering on MIXED run with {BASE_MODELS}...")
for mw in MOVING_WINDOWS:
    # Data partition and sample extraction
    partitions = mixed.mx_partition_data(
        terr_dfs,
        summary,
        mw,
        N_FOLDS,
        random_state=RANDOM_STATE,
    )

    print(f"Training models for a sampling window of {mw} seconds")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
