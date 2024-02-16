import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def performance_metrics_lineplot(results):
    marker_size = 12
    linewidth = 2
    fontsize = 24
    minperc = 40
    percres1 = 10
    percres2 = 5

    RN = list(results.keys())  # Results Names
    MN = RN[:-2]  # Model Names

    # Line Colors
    LCS = [
        "#0072BD",
        "#D95319",
        "#EDB120",
        "#7E2F8E",
        "#77AC30",
        "#4DBEEE",
        "#A2142F",
    ]

    # Markers
    MKS = ["d", "s", "^", "v", "h", "o", "+"]

    # FIGURE 1: ACCURACY PLOT
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title("ACCURACY", fontsize=fontsize + 10)
    ax1.set_xlabel("Sample window of analyzed terrain [s]", fontsize=fontsize)
    ax1.set_ylabel("Model accuracy [%]", fontsize=fontsize)
    ax1.grid(True)

    # analyze the confusion matrix for every sampling window and every model
    # store accuracy results and sampling window length
    for i, model_name in enumerate(MN):
        WN = list(results[model_name].keys())  # Window Names
        AT = np.zeros(len(WN))  # Accuracy Trend initialization
        WL = np.zeros(len(WN))  # Window Length initialization
        for j, window_name in enumerate(WN):
            CM = results[model_name][window_name]["ConfusionMat"]
            AT[j], _, _, _ = confusion_matrix_metrics(CM)
            win_idx = window_name.split("_")[1][:-2]
            # window length in seconds
            WL[j] = float(win_idx) / 1000

        ax1.plot(
            WL,
            AT,
            color=LCS[i],
            marker=MKS[i],
            markersize=marker_size,
            markerfacecolor=LCS[i],
            markeredgecolor=LCS[i],
            linewidth=linewidth,
            label=MN[i],
        )

    ax1.legend(fontsize=fontsize, loc="upper left")
    ax1.set_yticks(np.arange(0, 101, percres1))
    ax1.set_ylim([minperc, 100])
    ax1.set_xticks(WL)
    ax1.set_xlim([min(WL), max(WL)])
    ax1.axhline(
        y=minperc + percres2,
        linestyle="--",
        color="k",
        linewidth=2,
    )
    ax1.axhline(y=minperc, linestyle="-", color="k", linewidth=2)
    ax1.grid(which="both", linestyle="--", alpha=0.5)

    plt.show()

    # TODO: Other part of the function


def confusion_matrix_metrics(confusion_matrix):
    """
    Compute Accuracy, Sensitivity, Precision, and F1-score from a confusion matrix.

    Parameters:
    - confusion_matrix: numpy array, shape (num_classes, num_classes)
      Confusion matrix where rows represent true classes and columns represent predicted classes.

    Returns:
    - Accuracy: float
      Accuracy in percentage.
    - Sensitivity: numpy array, shape (num_classes,)
      Sensitivity for each class in percentage.
    - Precision: numpy array, shape (num_classes,)
      Precision for each class in percentage.
    - F1score: numpy array, shape (num_classes,)
      F1-score for each class in percentage.
    """
    # TODO: Compute differently
    total_samples = np.sum(confusion_matrix)
    accuracy = 100 * np.trace(confusion_matrix) / total_samples

    sensitivity = np.zeros(confusion_matrix.shape[0])
    for i in range(len(sensitivity)):
        sensitivity[i] = 100 * confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])

    precision = np.zeros(confusion_matrix.shape[0])
    for i in range(len(precision)):
        precision[i] = 100 * confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])

    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    return accuracy, sensitivity, precision, f1_score


def channels_plot(results):
    # Function to display the available channels of the results together with their labels.
    # The user can keep track of the active channels to which the following results correspond.
    fs = 20

    N_colors = 256
    vals = np.ones((N_colors, 4))
    vals[:, 0] = np.linspace(0.6350, 0.4660, N_colors)
    vals[:, 1] = np.linspace(0.0780, 0.6740, N_colors)
    vals[:, 2] = np.linspace(0.1840, 0.1880, N_colors)
    cmap = ListedColormap(vals)

    channels = results["Channels"]
    channel_names = list(channels.keys())

    fig, axs = plt.subplots(
        nrows=len(channel_names) + 1,
        ncols=1,
        sharex=True,
        gridspec_kw={"hspace": 0.5},
    )

    for channel, ax in zip(channel_names, axs):
        mat = np.array(channels[channel]["on"])[:, 1].astype(float).reshape(1, -1)
        # colors = np.pad(mat, [(0, 1), (0, 1)])

        rows = np.arange(mat.shape[1] + 1)
        cols = np.arange(mat.shape[0] + 1)

        r, c = np.meshgrid(rows, cols)
        pcm = ax.pcolormesh(r, c, mat, cmap=cmap)

        ax.set_xlim([rows[0], rows[-1]])
        ax.set_ylim([cols[0], cols[-1]])
        ax.set_xticks([])
        ax.set_yticks([])

        sf = channels[channel]["sf"]

        ax.set_title(
            f"Group identification: {channel}\nSampling frequency: {sf} [Hz]",
            fontsize=fs + 5,
        )

        xtxt = [(rows[i - 1] + rows[i]) / 2 for i in range(1, len(rows))]
        ytxt = np.full_like(xtxt, 0.5)

        features = channels[channel]["on"][:, 0]

        for x, y, txt in zip(xtxt, ytxt, features):
            ax.text(
                x,
                y,
                txt,
                ha="center",
                fontsize=fs,
            )

    ax = axs[-1]
    mat = np.array([0, 1]).astype(float).reshape(1, -1)
    colors = np.pad(mat, [(0, 1), (0, 1)])

    rows = np.arange(mat.shape[1] + 1) * 0.1
    cols = np.arange(mat.shape[0] + 1) * 0.1

    r, c = np.meshgrid(rows, cols)
    pcm = ax.pcolormesh(r, c, colors, cmap=cmap)
    ax.set_xlim([rows[0], rows[-1]])
    ax.set_ylim([cols[0], cols[-1]])
    ax.set_xticks([])
    ax.set_yticks([])

    xtxt = [(rows[i - 1] + rows[i]) / 2 for i in range(len(rows))]
    ytxt = np.full_like(xtxt, 0.5 * np.max(cols))
    texts = ["ON", "OFF"]
    for x, y, txt in zip(xtxt, ytxt, texts):
        ax.text(
            x,
            y,
            txt,
            ha="center",
            fontsize=fs,
        )

    ax.axis("equal")
    ax.set_facecolor("none")
    ax.axis("off")
    ax.set_title("Legend", fontsize=fs)

    fig.suptitle("Available sensor channels", fontsize=fs + 10)
    plt.show()
