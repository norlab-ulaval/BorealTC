from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as scio


def get_recordings_df(data_dir: Path, channels):
    """
    Extracts data from the specified data directory and returns a struct REC
    with two fields: data and time.

    Parameters:
    - data_dir: Path to the data directory.
    - channels: Struct containing channel information.

    Returns:
    - REC: Struct containing the numerical data and timestamps of the recordings.
    """
    terrains = [f.stem for f in data_dir.iterdir() if f.is_dir()]

    imu_cols = ["time", *channels["imu"]["cols"].keys()]
    pro_cols = ["time", *channels["pro"]["cols"].keys()]

    terr_dfs = {"imu": [], "pro": []}
    for terr in terrains:
        terr_dir = data_dir / terr

        imu_fnames = sorted([terr_dir.glob("imu_*.mat")])
        pro_fnames = sorted([terr_dir.glob("pro_*.mat")])

        imu_dfs = {}
        for i, fname in enumerate(imu_fnames):
            mat = scio.loadmat(fname)["imu"]
            df = pd.DataFrame(mat, columns=imu_cols)
            df["exp_idx"] = i
            imu_dfs[i] = df

        pro_dfs = {}
        for i, fname in enumerate(pro_fnames):
            mat = scio.loadmat(fname)["pro"]
            df = pd.DataFrame(mat, columns=pro_cols)
            df["exp_idx"] = i
            pro_dfs[i] = df

        terr_imu_df = pd.concat(imu_dfs.values(), ignore_index=True)
        terr_pro_df = pd.concat(pro_dfs.values(), ignore_index=True)
        terr_imu_df["terrain"] = terr
        terr_pro_df["terrain"] = terr

        terr_dfs.setdefault("imu", []).append(terr_imu_df)
        terr_dfs.setdefault("pro", []).append(terr_pro_df)

    # Filter columns ?
    return terr_dfs


def augment_data(train_dat, test_dat, summary, w, AUG):
    # Find the channel "c" providing data at higher frequency "sf" to be used
    # as a reference for windowing operation
    channel_names = summary.index.values
    print(channel_names)

    return

    c = 0
    sf = channels[channel_names[c]]["sf"]
    for i in range(1, len(channel_names)):
        if channels[channel_names[i]]["sf"] > sf:
            sf = channels[channel_names[i]]["sf"]
            c = i

    # Find the minimum sampling frequency channel
    FN = list(train_dat.keys())
    TotLabl = np.concatenate([train_dat[FN[0]]["labl"], test_dat[FN[0]]["labl"]])
    NumTer = int(np.max(TotLabl))
    SizTer = np.zeros(NumTer)

    for i in range(1, NumTer + 1):
        SizTer[i - 1] = np.sum(TotLabl == i)

    MinNum = int(np.min(SizTer))

    # Decide whether to augment the data homogeneously or not
    if AUG["same"] == 1:
        TotSmp = len(GenSmp) * MinNum
        TerSmp = np.floor(TotSmp / SizTer)
        TerSli = (1 / channels[channel_names[c]]["sf"]) * (
            (Samp.shape[0] - int(w * channels[channel_names[c]]["sf"])) / TerSmp
        )
    else:
        TerSli = np.full(NumTer, AUG["sliding_window"])

    # Augment the data using the appropriate sliding window for different
    # terrains or the same for every terrain depending on AUG.same
    AugTrain = {}
    AugTest = {}

    for i in range(Kfold):
        k = 0
        for j in range(len(train_dat[FN[i]]["data"])):
            sli = TerSli[int(train_dat[FN[i]]["labl"][j] - 1)]
            strt = 0
            stop = int(strt + w * channels[channel_names[c]]["sf"]) - 1
            while stop < train_dat[FN[i]]["data"][j][c].shape[0]:
                AugTrain[FN[i]]["data"][k][c] = train_dat[FN[i]]["data"][j][c][
                    strt:stop, :
                ]
                AugTrain[FN[i]]["time"][k][c] = train_dat[FN[i]]["time"][j][c][
                    strt:stop
                ]
                AugTrain[FN[i]]["labl"][k] = train_dat[FN[i]]["labl"][j]

                t0 = train_dat[FN[i]]["time"][j][c][strt]
                t1 = train_dat[FN[i]]["time"][j][c][stop]
                for s in range(len(channel_names)):
                    if s != c:
                        e0 = np.argmin(
                            np.abs(t0 - train_dat[FN[i]]["time"][j][channel_names[s]])
                        )
                        e1 = np.argmin(
                            np.abs(t1 - train_dat[FN[i]]["time"][j][channel_names[s]])
                        )
                        AugTrain[FN[i]]["data"][k][s] = train_dat[FN[i]]["data"][j][s][
                            e0:e1, :
                        ]
                        AugTrain[FN[i]]["time"][k][s] = train_dat[FN[i]]["time"][j][s][
                            e0:e1
                        ]
                        # Make the dimensions homogeneous
                        if AugTrain[FN[i]]["data"][k][s].shape[0] > int(
                            round(w * channels[channel_names[s]]["sf"])
                        ):
                            AugTrain[FN[i]]["data"][k][s] = np.delete(
                                AugTrain[FN[i]]["data"][k][s], -1, axis=0
                            )
                            AugTrain[FN[i]]["time"][k][s] = np.delete(
                                AugTrain[FN[i]]["time"][k][s], -1
                            )
                strt = int(strt + sli * channels[channel_names[c]]["sf"])
                stop = int(strt + w * channels[channel_names[c]]["sf"]) - 1
                k += 1

        k = 0
        for j in range(len(test_dat[FN[i]]["data"])):
            sli = TerSli[int(test_dat[FN[i]]["labl"][j] - 1)]
            strt = 0
            stop = int(strt + w * channels[channel_names[c]]["sf"]) - 1
            while stop < test_dat[FN[i]]["data"][j][c].shape[0]:
                AugTest[FN[i]]["data"][k][c] = test_dat[FN[i]]["data"][j][c][
                    strt:stop, :
                ]
                AugTest[FN[i]]["time"][k][c] = test_dat[FN[i]]["time"][j][c][strt:stop]
                AugTest[FN[i]]["labl"][k] = test_dat[FN[i]]["labl"][j]

                t0 = test_dat[FN[i]]["time"][j][c][strt]
                t1 = test_dat[FN[i]]["time"][j][c][stop]
                for s in range(len(channel_names)):
                    if s != c:
                        e0 = np.argmin(
                            np.abs(t0 - test_dat[FN[i]]["time"][j][channel_names[s]])
                        )
                        e1 = np.argmin(
                            np.abs(t1 - test_dat[FN[i]]["time"][j][channel_names[s]])
                        )
                        AugTest[FN[i]]["data"][k][s] = test_dat[FN[i]]["data"][j][s][
                            e0:e1, :
                        ]
                        AugTest[FN[i]]["time"][k][s] = test_dat[FN[i]]["time"][j][s][
                            e0:e1
                        ]
                        # Make the dimensions homogeneous
                        if AugTest[FN[i]]["data"][k][s].shape[0] > int(
                            round(w * channels[channel_names[s]]["sf"])
                        ):
                            AugTest[FN[i]]["data"][k][s] = np.delete(
                                AugTest[FN[i]]["data"][k][s], -1, axis=0
                            )
                            AugTest[FN[i]]["time"][k][s] = np.delete(
                                AugTest[FN[i]]["time"][k][s], -1
                            )
                strt = int(strt + sli * channels[channel_names[c]]["sf"])
                stop = int(strt + w * channels[channel_names[c]]["sf"]) - 1
                k += 1
