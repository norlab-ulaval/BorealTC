# %%
from pathlib import Path

import numpy as np
import pandas as pd
from utils import preprocessing, transforms
from scipy import stats

husky_data = Path("data/borealtc")
vulpi_data = Path("data/vulpi")

# %%
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

husky_summ = summary.copy()
vulpi_summ = summary.copy()

husky_dfs = preprocessing.get_recordings(husky_data, husky_summ)
vulpi_dfs = preprocessing.get_recordings(vulpi_data, vulpi_summ)

husky_pro = husky_dfs["pro"].copy()
vulpi_pro = vulpi_dfs["pro"].copy()

husky_pro["vx"], husky_pro["wz"] = transforms.unicycle_model(husky_pro)
vulpi_pro["vx"], vulpi_pro["wz"] = transforms.unicycle_model(vulpi_pro)

# %%
husk_metrics = {}
for terr in husky_pro.terrain.unique():
    terr_dat = husky_pro[husky_pro.terrain == terr]
    terr_metrics = {
        "iqr-wz": stats.iqr(terr_dat.wz.abs()),
        "med-wz": np.median(terr_dat.wz.abs()),
        "iqr-vx": stats.iqr(terr_dat.vx.abs()),
        "med-vx": np.median(terr_dat.vx.abs()),
    }
    husk_metrics[terr] = terr_metrics

vulp_metrics = {}
for terr in vulpi_pro.terrain.unique():
    terr_dat = vulpi_pro[vulpi_pro.terrain == terr]
    terr_metrics = {
        "iqr-wz": stats.iqr(terr_dat.wz.abs()),
        "med-wz": np.median(terr_dat.wz.abs()),
        "iqr-vx": stats.iqr(terr_dat.vx.abs()),
        "med-vx": np.median(terr_dat.vx.abs()),
    }
    vulp_metrics[terr] = terr_metrics

# %%
with open("husky-stats.dat", "w", encoding="utf-8") as f:
    for terr, m in husk_metrics.items():
        t = terr[:3]
        print(f"iw-{t} = {m['iqr-wz']:.2f}", file=f)
        print(f"mw-{t} = {m['med-wz']:.2f}", file=f)
        print(f"iv-{t} = {m['iqr-vx']:.2f}", file=f)
        print(f"mv-{t} = {m['med-vx']:.2f}", file=f)

with open("vulpi-stats.dat", "w", encoding="utf-8") as f:
    for terr, m in vulp_metrics.items():
        t = terr[:3]
        print(f"iw-{t} = {m['iqr-wz']:.2f}", file=f)
        print(f"mw-{t} = {m['med-wz']:.2f}", file=f)
        print(f"iv-{t} = {m['iqr-vx']:.2f}", file=f)
        print(f"mv-{t} = {m['med-vx']:.2f}", file=f)

# %%
