"""The SMS dataset."""

__all__ = [
    "SMSdataset"
]

import pandas as pd, os

class SMSdataset:
    """The SMS dataset.

    A label is binary, equals to 1 (0) if the message is a spam (ham).
    """

    def __init__(self, dataDir: str):
        self.dataDir = dataDir
        self.data = None

    def load(self):
        """Loads the sms dataset stored as a `pandas.DataFrame`"""
        path = os.path.join(
            self.dataDir, "SMSSpamCollection"
        )
        dataset = {"label": [], "text": []}
        with open(path, 'r') as f:
            for line in f.readlines():
                label, text = line.rstrip('\n').split('\t')
                dataset["label"].append(label)
                dataset["text"].append(text)
        self.data = pd.DataFrame(dataset)
        self.data["label"] = (self.data["label"] == "spam").astype(int)

    def head(self, *args, **kwargs):
        return self.data.head(*args, **kwargs)
