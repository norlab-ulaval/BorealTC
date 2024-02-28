from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

res_dir = Path("results")
met_dir = Path("metrics")
met_dir.mkdir(exist_ok=True, parents=True)


def p(n: float, factor: int = 100) -> float:
    return factor * n


def process_results(res_path: Path):
    dataset = res_path.parent.stem
    if dataset not in ["baseline", "husky", "vulpi", "combined"]:
        return
    # TODO: Add augmented data
    elems = res_path.stem.split("_")
    _, model, desc, _, mw, *_ = elems
    win = int(p(float(mw), 1000))

    values = np.load(res_path, allow_pickle=True).item()

    terrains = np.array([t for t in values["terrains"] if t != "MIXED"])

    idxpred = values["pred"]
    idxtest = values["true"]

    if idxpred.dtype.type is np.int64:
        ypred = terrains[idxpred]
        ytest = terrains[idxtest]
    else:
        ypred = idxpred
        ytest = idxtest

    terr_idx = {t: i for i, t in enumerate(terrains)}

    acc = accuracy_score(
        ytest,
        ypred,
    )
    prec = precision_score(
        ytest,
        ypred,
        labels=terrains,
        average=None,
    )
    reca = recall_score(
        ytest,
        ypred,
        labels=terrains,
        average=None,
    )
    f1 = f1_score(
        ytest,
        ypred,
        labels=terrains,
        average=None,
    )
    ap = average_precision_score(
        np.array([terr_idx[y] for y in ytest]).reshape(-1, 1),
        np.array([terr_idx[y] for y in ypred]).reshape(-1, 1),
        average=None,
    ).item()

    metdat = met_dir / dataset
    metdat.mkdir(parents=True, exist_ok=True)
    fname = metdat / f"{model}-{win}-{desc}.dat"
    with open(fname, "w", encoding="utf-8") as f:
        print(f"acc = {p(acc):.2f}", file=f)
        print(f"ap = {p(ap):.2f}", file=f)
        for idx, terr in enumerate(terrains):
            t = terr[:3]
            print(f"p-{t} = {p(prec[idx]):.2f}", file=f)
            print(f"r-{t} = {p(reca[idx]):.2f}", file=f)
            print(f"f-{t} = {p(f1[idx]):.2f}", file=f)


def baseline_export(
    cfmtx: np.ndarray,
    model: str,
    win: float,
    terrains: list[str],
    metrics_dir: Path,
):
    metrics_dir.mkdir(parents=True, exist_ok=True)

    acc = np.trace(cfmtx) / cfmtx.sum()
    prec = cfmtx.diagonal() / cfmtx.sum(axis=0)
    reca = cfmtx.diagonal() / cfmtx.sum(axis=1)
    f1 = 2 * prec * reca / (prec + reca)

    fname = metrics_dir / f"{model}-{win}.dat"
    with open(fname, "w", encoding="utf-8") as f:
        print(f"acc = {p(acc):.2f}", file=f)
        for idx, terr in enumerate(terrains):
            t = terr[:3]
            print(f"p-{t} = {p(prec[idx]):.2f}", file=f)
            print(f"r-{t} = {p(reca[idx]):.2f}", file=f)
            print(f"f-{t} = {p(f1[idx]):.2f}", file=f)


def format_win(s: str) -> str:
    return s.split("_")[1][:-2]


def process_baseline(res_fname: str, metrics_dir: Path):
    baseline = scio.loadmat(res_dir / res_fname, matlab_compatible=True)["RES"]

    labels = baseline["TerLabls"].item()
    n_labels = labels.shape[1]

    terrains = ["".join(labels[0, i].flatten().tolist()) for i in range(n_labels)]

    models = ["CNN", "LSTM", "CLSTM", "SVM"]
    for mod in models:
        modres = baseline[mod].item()
        for win in modres.dtype.names:
            modwin = modres[win].item()
            fwin = int(format_win(win))
            cfmtx = modwin["ConfusionMat"].item()
            baseline_export(cfmtx, mod, fwin, terrains, metrics_dir)


def main():
    res_paths = filter(lambda x: len(x.parents) > 2, res_dir.rglob("**/*.npy"))
    for res in tqdm(res_paths):
        process_results(res_path=res)

    process_baseline("TDEEP.mat", met_dir / "baseline-vulpi")
    # process_baseline("TDEEP-norlab.mat", met_dir / "baseline-husky")


if __name__ == "__main__":
    main()
