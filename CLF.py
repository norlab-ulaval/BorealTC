from itertools import chain
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from utils import preprocessing

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

csv_dir = Path("norlab-data")

X_cols = [[k for k, v in ch.items() if v] for ch in columns.values()]
X_cols = list(chain.from_iterable(X_cols))

# Get recordings
data = preprocessing.get_recordings(csv_dir, summary)
merged = preprocessing.merge_upsample(data, summary, mode="last")

X = merged[X_cols].copy()
y = merged.terrain.copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3)

print("=== RF ===")

rf_clf = RandomForestClassifier(n_estimators=500, oob_score=True)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)
print(rf_clf.score(X_test, y_test))

print("=== MLP ===")

rf_clf = MLPClassifier(max_iter=750)
rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_test)
print(rf_clf.score(X_test, y_test))
