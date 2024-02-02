def partition_data_csv(
    data: ExperimentData,
    summary: pd.DataFrame,
    partition_duration: float,
    n_splits: int = 5,
    random_state: int | None = None,
):
    # Highest sampling frequency
    hf_sensor = summary["sampling_freq"].idxmax()
    hf = summary["sampling_freq"].max()
    # Other sensors are low frequency
    lf_sensors = tuple(sens for sens in data.keys() if sens != hf_sensor)

    # Time (s) / window * Sampling freq = samples / window
    wind_len = int(partition_duration * hf)

    # Create partition windows
    partitions = {hf_sensor: {}, **{s: {} for s in lf_sensors}}

    # Data from the high frequency sensor
    hf_data = data[hf_sensor]
    terrains = hf_data.terrain.unique().tolist()

    for terr_idx, terr in enumerate(terrains):
        hf_terr = hf_data[hf_data.terrain == terr].assign(terr_idx=terr_idx)
        exp_idxs = sorted(hf_terr.run_idx.unique())
        for exp_idx in exp_idxs:
            hf_exp = hf_terr[hf_terr.run_idx == exp_idx].copy().reset_index(drop=True)

            # Get limits, avoid selecting incomplete partitions
            starts = np.arange(0, hf_exp.shape[0], wind_len)
            starts = starts[(starts + wind_len) <= hf_exp.shape[0]]
            limits = np.vstack([starts, starts + wind_len]).T

            # Get multiple windows
            windows = [
                hf_exp.iloc[slice(*lim)].assign(win_idx=win_idx)
                for win_idx, lim in enumerate(limits)
            ]
            hf_cols = windows[0].columns.tolist()
            hf_c = [*np.take(hf_cols, (-4, -2, -3, -1, 0)), *hf_cols[1:-4]]
            windows = [w[hf_c] for w in windows]
            tlimits = [np.array([w.time.min(), w.time.max()]) for w in windows]
            partitions[hf_sensor].setdefault(terr, []).extend(windows)

            # Slice each lf sensor based on the time from the hf windows
            for sens in lf_sensors:
                lf_data = data[sens]
                lf_terr = lf_data[lf_data.terrain == terr].assign(terr_idx=terr_idx)
                lf_exp = lf_terr[lf_terr.run_idx == exp_idx]
                lf_exp = lf_exp.copy().reset_index(drop=True)
                lf_time = lf_exp.time.to_numpy()[None, :]

                indices = np.array(
                    [np.abs(lf_time - tlim[:, None]).argmin(axis=1) for tlim in tlimits]
                )
                indices[:, 1] += 1
                lf_win = [
                    lf_exp.iloc[slice(*lim)].assign(win_idx=win_idx)
                    for win_idx, lim in enumerate(indices)
                ]
                lf_cols = lf_win[0].columns.tolist()
                lf_c = [*np.take(lf_cols, (-4, -2, -3, -1, 0)), *lf_cols[1:-4]]
                lf_win = [w[lf_c] for w in lf_win]
                partitions[sens].setdefault(terr, []).extend(lf_win)

    hf_columns = partitions[hf_sensor][terrains[0]][0].columns.values
    terr_col = np.where(hf_columns == "terrain")
    # Number partitions x time x channels
    # terrain, terr_idx, run_idx, win_idx, time, <sensor_channels>
    unified = {
        sens: np.vstack([sens_data[terr] for terr in terrains])
        for sens, sens_data in partitions.items()
    }
    n_windows = unified[hf_sensor].shape[0]
    labels = unified[hf_sensor][:, 0, terr_col][:, 0, 0]
    # for sens, sens_data in unified.items():
    #     print(sens, sens_data.shape, (sens_data[:, 0, :][:, 0] == labels).all())

    # Split with K folds
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    train_data, test_data = [], []

    for fold_train_idx, fold_test_idx in skf.split(np.zeros(len(labels)), labels):
        train_data.append(
            {
                sens: sens_data[fold_train_idx, :, :]
                for sens, sens_data in unified.items()
            }
        )
        test_data.append(
            {
                sens: sens_data[fold_test_idx, :, :]
                for sens, sens_data in unified.items()
            }
        )

    return train_data, test_data
