import unittest
from pathlib import Path

import pandas as pd
import scipy.io as scio


class TestPreprocessing(unittest.TestCase):
    tests_dir = Path("tests")

    def setUp(self) -> None:
        self.summary = pd.read_csv(self.tests_dir / "summary.csv")
        mat_paths = self.tests_dir.rglob("*.mat")
        mat_files = [scio.loadmat(p, matlab_compatible=True) for p in mat_paths]
        self.mat = mat_files

    def test_augmentation(self):
        fold_name = "folder_1"
        for mat_file in self.mat:
            train = mat_file["Train"][fold_name].item()
            test = mat_file["Test"][fold_name].item()
            aug_train = mat_file["AugTrain"][fold_name].item()
            aug_test = mat_file["AugTest"][fold_name].item()
            for val in (train, test, aug_train, aug_test):
                print(val["data"].item().shape, val["time"].item().shape)
        print(self.summary.head())
        assert sum([1, 2, 3]) == 6
        assert sum([1, 2, 3]) == 9


if __name__ == "__main__":
    unittest.main()
