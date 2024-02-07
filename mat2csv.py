from __future__ import annotations

from pathlib import Path

import pandas as pd
import scipy.io as scio

mat_dir = Path("datasets")
csv_dir = Path("data")

cols = {
    "imu": [
        "wx",
        "wy",
        "wz",
        "ax",
        "ay",
        "az",
    ],
    "pro": [
        "velL",
        "velR",
        "curL",
        "curR",
    ],
}


def mat_to_csv(fname: Path) -> Path:
    terr = fname.parent.stem

    datatype = fname.stem.split("_")[0]
    data = scio.loadmat(fname, matlab_compatible=True)[datatype]
    datacols = ["time", *cols[datatype]]

    df = pd.DataFrame(data, columns=datacols)

    csv_terr_dir = csv_dir / terr
    csv_terr_dir.mkdir(exist_ok=True, parents=True)

    export_path = (csv_terr_dir / fname.stem).with_suffix(".csv")
    df.to_csv(export_path, index=False)

    return export_path


if __name__ == "__main__":
    files = [*mat_dir.rglob("*.mat")]
    for f in files:
        mat_to_csv(f)
