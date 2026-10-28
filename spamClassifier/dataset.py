"""The SMS dataset."""

__all__ = [
    "loadSMSdataset"
]

import pandas as pd, os

def loadSMSdataset(dataDir: str):
    """Returns the sms dataset stored as a `pandas.DataFrame`.
    
    A label is binary, equals to 1 (0) if the message is a spam (ham).
    """
    path = os.path.join(
        dataDir, "SMSSpamCollection"
    )
    dataset = {"label": [], "text": []}
    with open(path, 'r') as f:
        for line in f.readlines():
            label, text = line.rstrip('\n').split('\t')
            dataset["label"].append(label)
            dataset["text"].append(text)
    df = pd.DataFrame(dataset)
    df["label"] = (df["label"] == "spam").astype(int)
    return df
