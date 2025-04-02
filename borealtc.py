import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class BorealTCRecord:
    imu_path: pathlib.Path
    pro_path: pathlib.Path
    class_name: str
    run_id: str


@dataclass
class BorealTCSample:
    imu_df: pd.DataFrame
    pro_df: pd.DataFrame
    class_name: str
    run_id: str


@dataclass
class BorealTCFusedSample:
    fused_df: pd.DataFrame
    class_name: str
    run_id: str


class BorealTC(Dataset):
    def __init__(self, root: str, transform=None, classes: Optional[list[str]] = None):
        """
        BorealTC dataset.
        :param root: root directory containing the dataset
        :param transform: Optional transform to be applied on a sample.
        :param classes: Optional list of classes to load. If None, all classes are loaded.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.data_per_class = defaultdict(dict)
        self.columns = [
            "wx",
            "wy",
            "wz",
            "ax",
            "ay",
            "az",
            "curL",
            "curR",
            "velL",
            "velR",
        ]

        class_paths = sorted(
            [d for d in self.root.iterdir() if d.is_dir() and d.stem != "MIXED"]
        )
        self.classes = classes if classes else [d.stem.lower() for d in class_paths]
        self.class_to_idx = {k: i for i, k in enumerate(self.classes)}

        self.data_records = []
        for class_path in class_paths:
            class_name = class_path.stem.lower()
            if self.classes and class_name not in self.classes:
                continue
            for imu_path in sorted(class_path.glob("imu_*.csv")):
                run_id = imu_path.stem.split("_")[1]
                pro_path = class_path / f"pro_{run_id}.csv"
                record = BorealTCRecord(imu_path, pro_path, class_name, run_id)
                self.data_records.append(record)

    def __len__(self) -> int:
        return len(self.data_records)

    def __getitem__(self, idx) -> BorealTCFusedSample:
        record = self.data_records[idx]
        imu_df = pd.read_csv(record.imu_path).set_index("time")
        pro_df = pd.read_csv(record.pro_path).set_index("time")
        imu_df.index = pd.to_timedelta(imu_df.index, unit="s")
        pro_df.index = pd.to_timedelta(pro_df.index, unit="s")
        fused = fuse_measures(imu_df, pro_df, self.columns)
        sample = BorealTCFusedSample(fused, record.class_name, record.run_id)
        if self.transform:
            sample = self.transform(sample)
        return sample


def fuse_measures(imu_df, pro_df, cols):
    """
    Fuses the IMU and PRO dataframes by aligning them on the time index of the highest frequency data.
    :param imu_df: IMU dataframe
    :param pro_df: proprioception dataframe
    :param cols: columns to keep
    :return: Fused dataframe
    """
    freq1 = pd.infer_freq(imu_df.index)
    freq2 = pd.infer_freq(pro_df.index)
    highest_freq = min(freq1, freq2) if freq1 and freq2 else freq1 or freq2

    imu_df = imu_df.resample(highest_freq).ffill()
    pro_df = pro_df.resample(highest_freq).ffill()

    aligned = pd.concat([imu_df, pro_df], axis=1).ffill()

    return aligned[cols]


class SlidingWindowDataset(Dataset):
    """Generates sliding windows from the BorealTC fused dataset."""

    def __init__(self, dataset: BorealTC, window_size: int = 170, step_size: int = 50):
        """
        Args:
            dataset: Sliding windows dataset from BorealTC
            window_size: Number of time steps in each window, default is 170 which corresponds to 1.7 seconds
            step_size: Step size between windows, default is 50 which corresponds to 0.5 seconds
        """
        self.dataset = dataset
        self.window_size = window_size
        self.step_size = step_size
        self.windows = self._generate_windows()

    def _generate_windows(self):
        """Precomputes the start and end indices of all windows"""
        windows = []
        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            df = sample.fused_df

            total_steps = len(df)
            for start in range(0, total_steps - self.window_size + 1, self.step_size):
                end = start + self.window_size
                windows.append((idx, start, end))

        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sample_idx, start, end = self.windows[idx]
        sample = self.dataset[sample_idx]
        window_df = sample.fused_df.iloc[start:end]
        window_tensor = torch.tensor(window_df.values, dtype=torch.float32)

        return {
            "window": window_tensor,
            "class_name": sample.class_name,
            "run_id": sample.run_id,
        }


if __name__ == "__main__":
    # Load the full dataset
    dataset = BorealTC("data/borealtc")

    # Create sliding window dataset
    window_dataset = SlidingWindowDataset(dataset, window_size=170, step_size=50)

    # Iterate through the sliding window dataset
    for i in range(len(window_dataset)):
        sample = window_dataset[i]
        print(
            f"Window {i}: {sample['window'].shape}, Class: {sample['class_name']}, Run ID: {sample['run_id']}"
        )
