import pathlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
)

path = pathlib.Path("results/split")

values = np.load("results/data_concat/results_hamming_CNN_mw_1.7.npy", allow_pickle=True).item()
terrains = np.array([t for t in values["terrains"] if t != "MIXED"])
inv_terrains = {t: i for i, t in enumerate(terrains)}


def convert_to_int(x):
    if x.dtype == str:
        return np.array([inv_terrains[t] for t in x])
    return x


res_cnn = []
res_mamba = []
for p in sorted(path.iterdir()):
    if 'CNN' in p.stem:
        res_cnn.append(np.load(p, allow_pickle=True).item())
    elif 'mamba' in p.stem:
        res_mamba.append(np.load(p, allow_pickle=True).item())

terrains = np.array([t for t in values["terrains"] if t != "MIXED"])
inv_terrains = {t: i for i, t in enumerate(terrains)}

cnn_acc_per_split = []
mamba_acc_per_split = []
dim_splits = []
for res in res_cnn[::-1]:
    ypred = convert_to_int(res["pred"])
    ytest = convert_to_int(res["true"])
    fold_size = len(ypred) // 5
    print(fold_size)
    dim_splits.append(fold_size)
    acc = []
    for k in range(5):
        pred = ypred[k * fold_size: (k + 1) * fold_size]
        true = ytest[k * fold_size: (k + 1) * fold_size]
        acc.append(accuracy_score(true, pred))
    cnn_acc_per_split.append(acc)

for res in res_mamba[::-1]:
    ypred = convert_to_int(res["pred"])
    ytest = convert_to_int(res["true"])
    fold_size = len(ypred) // 5
    print(fold_size)
    acc = []
    for k in range(5):
        pred = ypred[k * fold_size: (k + 1) * fold_size]
        true = ytest[k * fold_size: (k + 1) * fold_size]
        acc.append(accuracy_score(true, pred))
    mamba_acc_per_split.append(acc)

print(f'{cnn_acc_per_split=}')
print(f'{mamba_acc_per_split=}')

x = np.arange(1, 6)
# x = dim_splits
# print(x)

cnn_acc = np.mean(cnn_acc_per_split, axis=1)
mamba_acc = np.mean(mamba_acc_per_split, axis=1)

# cnn_bars = []
# mamba_bars = []
# fq75, q25 = np.percentile(x, [75 ,25])or i in range(5):
#     cnn_bars.append([np.min(cnn_acc_per_split[i]), np.max(cnn_acc_per_split[i])])
#     mamba_bars.append([np.min(mamba_acc_per_split[i]), np.max(mamba_acc_per_split[i])])

cnn_bars = np.percentile(cnn_acc_per_split, [75, 25], axis=1).T
mamba_bars = np.percentile(mamba_acc_per_split, [75, 25], axis=1).T
print(cnn_bars.shape)
print(mamba_bars.shape)
# cnn_bars = np.array(cnn_bars)
# mamba_bars = np.array(mamba_bars)

plt.plot(x, cnn_acc, label='cnn', marker='o')
plt.plot(x, mamba_acc, label='mamba', marker='o')

plt.fill_between(x, cnn_bars[:, 0], cnn_bars[:, 1], alpha=0.1)
plt.fill_between(x, mamba_bars[:, 0], mamba_bars[:, 1], alpha=0.1)

# plt.xscale('log')
# plt.yscale('log')

# plt.errorbar(x, cnn_acc, yerr=cnn_bars, fmt='-o', label='cnn')
# plt.errorbar(x, mamba_acc, yerr=mamba_bars, fmt='-o', label='mamba')

plt.legend(loc='best')
plt.show()
