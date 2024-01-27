from pathlib import Path

import matplotlib.pyplot as plt
import scipy.io as sio

from utils import analysis

cwd = Path.cwd()
data_dir = cwd / "datasets"
results_dir = cwd / "results"

# Load result file
result_name = "TrainingResults_1"
results = sio.loadmat(results_dir / f"{result_name}.mat", simplify_cells=True)["RES"]

# Section 2: Load and Visualize Channels
# analysis.channels_plot(results)

# Section 3: Performance Metrics Line Plots
analysis.performance_metrics_lineplot(results)

# # Section 4: Confusion Matrix Plot
# CMprop = {
#     "FontSize": 10,
#     "Normalization": "row-normalized",
#     "FixedSampWind": 1.7,  # Choose a specific sampling window
# }
# ConfusionMatrix_Plot(RES, CMprop)

plt.show()
