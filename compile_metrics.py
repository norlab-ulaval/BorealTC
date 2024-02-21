from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

res_dir = Path("results")
met_dir = Path("metrics")
met_dir.mkdir(exist_ok=True, parents=True)


def p(n: float, factor: int = 100) -> float:
    return factor * n


def process_results(res_path: Path):
    dataset = res_path.parent.stem
    _, model, _, mw, *_ = res_path.stem.split("_")
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

    metdat = met_dir / dataset
    metdat.mkdir(parents=True, exist_ok=True)
    fname = metdat / f"{model}-{win}.dat"
    with open(fname, "w", encoding="utf-8") as f:
        print(f"acc = {p(acc):.2f}", file=f)
        for idx, terr in enumerate(terrains):
            t = terr[:3]
            print(f"p-{t} = {p(prec[idx]):.2f}", file=f)
            print(f"r-{t} = {p(reca[idx]):.2f}", file=f)
            print(f"f-{t} = {p(f1[idx]):.2f}", file=f)


def baseline_export(cfmtx: np.ndarray, model: str, win: float, terrains: list[str]):
    metbas = met_dir / "baseline"
    metbas.mkdir(parents=True, exist_ok=True)

    acc = np.trace(cfmtx) / cfmtx.sum()
    prec = cfmtx.diagonal() / cfmtx.sum(axis=0)
    reca = cfmtx.diagonal() / cfmtx.sum(axis=1)
    f1 = 2 * prec * reca / (prec + reca)

    fname = metbas / f"{model}-{win}.dat"
    with open(fname, "w", encoding="utf-8") as f:
        print(f"acc = {p(acc):.2f}", file=f)
        for idx, terr in enumerate(terrains):
            t = terr[:3]
            print(f"p-{t} = {p(prec[idx]):.2f}", file=f)
            print(f"r-{t} = {p(reca[idx]):.2f}", file=f)
            print(f"f-{t} = {p(f1[idx]):.2f}", file=f)


def format_win(s: str) -> str:
    return s.split("_")[1][:-2]


def process_baseline():
    baseline = scio.loadmat(res_dir / "TDEEP.mat", matlab_compatible=True)["RES"]

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
            baseline_export(cfmtx, mod, fwin, terrains)


def main():
    res_paths = filter(lambda x: len(x.parents) > 2, res_dir.rglob("**/*.npy"))
    for res in res_paths:
        process_results(res_path=res)

    process_baseline()


if __name__ == "__main__":
    main()
