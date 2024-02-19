from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pipeline as pp
import scipy
from torch.utils.data import Dataset

from utils.constants import ch_cols


@dataclass
class VulpiData:
    imu: np.ndarray
    pro: np.ndarray
    label: str
    label_id: int
    run_id: int
    imu_path: str
    pro_path: str


class RawVulpiDataset(Dataset):
    def __init__(self, root_dir: Path, transform: Optional[Callable] = None):
        self._root_dir = root_dir
        self._transform = transform
        self._class_paths = sorted([x for x in root_dir.iterdir() if x.is_dir()])
        self._class_names = [x.name for x in self._class_paths]
        self._class_name_to_id = {x: i for i, x in enumerate(self._class_names)}
        self._id_to_class_name = {v: k for k, v in self._class_name_to_id.items()}
        imus = self._root_dir.rglob("imu_*.mat")
        pros = self._root_dir.rglob("pro_*.mat")
        self._imu_paths = sorted(imus)
        self._pro_paths = sorted(pros)

    def __len__(self):
        return len(self._imu_paths)

    def __getitem__(self, idx):
        imu_path = self._imu_paths[idx]
        imu = scipy.io.loadmat(imu_path)
        pro_path = self._pro_paths[idx]

        pro = scipy.io.loadmat(pro_path)
        label = imu_path.parent.name
        label_id = self._class_name_to_id[label]
        run_id = int(imu_path.name.split("_")[1].split(".")[0])

        data = VulpiData(
            imu=imu["imu"],
            pro=pro["pro"],
            label=label,
            label_id=label_id,
            run_id=run_id,
            imu_path=imu_path,
            pro_path=pro_path,
        )
        if self._transform is not None:
            data = self._transform(data)

        return data

    @property
    def class_to_id(self):
        return self._class_name_to_id

    @property
    def id_to_class(self):
        return self._id_to_class_name


class TemporalDataset(Dataset):
    def __init__(self, data, transform: Optional[Callable] = None):
        self.data = data
        self.transform = transform if transform is not None else pp.Identity()

    def __len__(self):
        return len(self.data["imu"])

    def __getitem__(self, idx):
        imu = self.data["imu"][idx]
        pro = self.data["pro"][idx]
        label = imu[:, ch_cols["terr_idx"]][0]

        imu_channels = imu[:, 5:]
        pro_channels = pro[:, 5:]

        sample = dict(imu=imu_channels, pro=pro_channels), label
        return self.transform(sample)


class MCSDataset(Dataset):
    def __init__(self, mcs, transform: Optional[Callable] = None):
        super().__init__()
        self.mcs = mcs
        self.transform = transform if transform is not None else pp.Identity()

    def __len__(self):
        return len(self.mcs["data"])

    def __getitem__(self, idx):
        sample = self.mcs["data"][idx], self.mcs["label"][idx]
        return self.transform(sample)


class MambaDataset(Dataset):
    def __init__(self, data, transform: Optional[Callable] = None):
        super().__init__()
        self.data = data
        self.transform = transform if transform is not None else pp.Identity()

    def __len__(self):
        return len(self.data["imu"]["data"])

    def __getitem__(self, idx):
        sample = (
            dict(
                imu=self.data["imu"]["data"][idx, :],
                pro=self.data["pro"]["data"][idx, :],
                # imu_data=self.data['imu']['data'][idx, :, 1:],
                # imu_time=self.data['imu']['data'][idx, :, 0],
                # pro_data=self.data['pro']['data'][idx, :, 1:],
                # pro_time=self.data['pro']['data'][idx, :, 0],
            ),
            self.data["imu"]["label"][idx],
        )

        return self.transform(sample)
