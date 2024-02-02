import numpy as np


def augment_data(Train, Test, Channels, w, AUG):
    def generate_samples(data, time, label, sliding_window):
        start, stop = 0, int(w * hf)
        while stop <= data.shape[0]:
            yield {
                "data": data[start:stop, :],
                "time": time[start:stop],
                "label": label,
            }
            start += int(sliding_window * hf)
            stop = start + int(w * hf)

    def augment_dataset(dataset):
        augmented_dataset = {"data": {}, "time": {}, "label": []}
        for j, (data_j_c, time_j_c, label_j) in enumerate(
            zip(dataset["data"][c], dataset["time"][c], dataset["label"])
        ):
            sliding_window = (
                AUG["sliding_window"]
                if not AUG["same"]
                else AUG["sliding_window"][label_j - 1]
            )
            k = 1
            for sample in generate_samples(data_j_c, time_j_c, label_j, sliding_window):
                augmented_dataset["data"][k] = {c: sample["data"]}
                augmented_dataset["time"][k] = {c: sample["time"]}
                augmented_dataset["label"].append(sample["label"])
                for s in CN:
                    if s != c:
                        indices = np.searchsorted(
                            dataset["time"][j][s],
                            [sample["time"][0], sample["time"][-1]],
                        )
                        augmented_dataset["data"][k][s] = dataset["data"][j][s][
                            indices[0] : indices[1], :
                        ]
                        augmented_dataset["time"][k][s] = dataset["time"][j][s][
                            indices[0] : indices[1]
                        ]
                        # make the dimensions homogeneous
                        augmented_dataset["data"][k][s] = (
                            augmented_dataset["data"][k][s][:-1, :]
                            if augmented_dataset["data"][k][s].shape[0]
                            > round(w * Channels[s]["sf"])
                            else augmented_dataset["data"][k][s]
                        )
                        augmented_dataset["time"][k][s] = (
                            augmented_dataset["time"][k][s][:-1]
                            if augmented_dataset["time"][k][s].shape[0]
                            > round(w * Channels[s]["sf"])
                            else augmented_dataset["time"][k][s]
                        )
                k += 1
        return augmented_dataset

    AugTrain = {fold: augment_dataset(Train[fold]) for fold in Train}
    AugTest = {fold: augment_dataset(Test[fold]) for fold in Test}

    return AugTrain, AugTest


# Example usage:
# AugTrain, AugTest = augment_data(Train, Test, Channels, w, AUG)
