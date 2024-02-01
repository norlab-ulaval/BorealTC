from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import scipy
from torch.utils.data import Dataset


@dataclass
class VulpiData:
    imu: np.ndarray
    pro: np.ndarray
    label: str
    label_id: int
    run_id: int
    imu_path: str
    pro_path: str


class VulpiDataset(Dataset):
    def __init__(self, root_dir: Path, transform: Optional[Callable] = None):
        self._root_dir = root_dir
        self._transform = transform
        self._class_paths = sorted([x for x in root_dir.iterdir() if x.is_dir()])
        self._class_names = [x.name for x in self._class_paths]
        self._class_name_to_id = {x: i for i, x in enumerate(self._class_names)}
        self._id_to_class_name = {v: k for k, v in self._class_name_to_id.items()}
        imus = self._root_dir.rglob('imu_*.mat')
        pros = self._root_dir.rglob('pro_*.mat')
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
        run_id = int(imu_path.name.split('_')[1].split('.')[0])

        data = VulpiData(imu=imu['imu'], pro=pro['pro'], label=label, label_id=label_id, run_id=run_id,
                         imu_path=imu_path, pro_path=pro_path)
        if self._transform is not None:
            data = self._transform(data)

        return data

    @property
    def class_to_id(self):
        return self._class_name_to_id

    @property
    def id_to_class(self):
        return self._id_to_class_name


if __name__ == '__main__':
    dataset = VulpiDataset(root_dir=Path('datasets'))
    for d in dataset:
        print(d.label_id, d.label, d.run_id, d.imu.shape, d.pro.shape)
        print(d.imu_path, d.pro_path)
