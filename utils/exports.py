import json
from pathlib import Path


class JSONExporter:
    def __init__(self, path):
        self.path = Path(path)
        self.data = None

    def __enter__(self):
        with open(self.path, mode="r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.data is not None:
            with open(self.path, mode="w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, sort_keys=True)
