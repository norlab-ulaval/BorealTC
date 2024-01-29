import numpy as np
from sklearn.model_selection import StratifiedKFold


def partition_data(
    REC, Channels, PW: int, n_splits: int = 5, random_state: int | None = None
):
    sensor_names = list(Channels.keys())
    max_freq_sensor = max(
        Channels, key=lambda k: Channels[k]["sf"]
    )  # Find the channel providing data at higher frequency
    max_freq = Channels[max_freq_sensor]["sf"]
    nb_pts_wind = int(PW * max_freq)

    PRT = {"data": {}, "time": {}}
    terrains = list(REC["data"].keys())

    for terr in terrains:
        PRT["time"].setdefault(terr, [])
        PRT["data"].setdefault(terr, [])
        runs_data = REC["data"][terr]
        runs_time = REC["time"][terr]
        for exp_time, exp_data in zip(runs_time, runs_data):
            data_c = exp_data[max_freq_sensor]
            time_c = exp_time[max_freq_sensor]

            starts = np.arange(0, len(data_c), nb_pts_wind)
            limits = np.vstack([starts, starts + nb_pts_wind]).T

            # For each window
            for lim in limits:
                PRT["time"][terr].append(time_c[slice(*lim)])
                PRT["data"][terr].append(data_c[slice(*lim), :])

                other_sens = [sens for sens in sensor_names if sens != max_freq_sensor]

                for sensor in other_sens:
                    indices = np.searchsorted(
                        exp_time[sensor],
                        [time_c[lim[0]], time_c[lim[1]]],
                    )
                    PRT["time"][terr].append(exp_time[sensor][indices[0] : indices[1]])
                    PRT["data"][terr].append(
                        exp_data[sensor][indices[0] : indices[1], :]
                    )

    UNF = {
        "data": np.vstack(PRT["data"][terr] for terr in terrains),
        "time": np.vstack(PRT["time"][terr] for terr in terrains),
        "labels": np.repeat(
            np.arange(len(terrains)), [len(PRT["data"][terr]) for terr in terrains]
        ),
    }

    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    train_data, test_data = {}, {}

    for i, (train_index, test_index) in enumerate(
        skf.split(np.zeros_like(UNF["labels"]), UNF["labels"])
    ):
        # For each fold
        train_data[f"fold_{i + 1}"] = {
            "data": UNF["data"][train_index],
            "time": UNF["time"][train_index],
            "labels": UNF["labels"][train_index],
        }
        test_data[f"fold_{i + 1}"] = {
            "data": UNF["data"][test_index],
            "time": UNF["time"][test_index],
            "labels": UNF["labels"][test_index],
        }

    return train_data, test_data


# Example usage:
# Train, Test = partition_data(REC, Channels, KFOLD, PART_WINDOW, RNG)
