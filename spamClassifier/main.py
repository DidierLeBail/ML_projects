import pandas as pd
import numpy as np
import os

datapath = os.path.join(
    os.path.dirname(__file__), "data", "SMSSpamCollection.txt"
)

# For some reason, this line did not work (5572 rows instead of 5574)
# dataset = pd.read_csv(datapath, sep='\t', names=("label", "text"))
# so i ended up loading the data with open into a dictionary

dataset = {"label": [], "text": []}
with open(datapath, 'r') as f:
    lines = f.readlines()
    for line in lines:
        label, text = line.rstrip('\n').split('\t')
        dataset["label"].append(label)
        dataset["text"].append(text)
dataset = pd.DataFrame(dataset)

print(dataset.head())
print(dataset.tail())
print()

print(f"nb of samples: {len(dataset.index)}, should be 5574")
assert len(dataset) == 5574
print()

class SpamClassifier:
    """A classifier that, given a SMS returns its probability of being a spam or a regular (ham) message.
    
    A SMS here is converted into a vector-valued time series (embedding).
    The multinomial Bayesian classification model will then be independent from the time ordering of these series.
    """

    def __init__(self):
        pass
    
    def fit(self, X_train: np.ndarray, Y_train: np.ndarray):
        """Fits the model parameters from the training examples `X_train` and
        their labels `Y_train`.
        """

    def __call__(self, msgs: np.ndarray):
        pass

def inspect_data(dataset: pd.DataFrame):
    """Computes some basic statistics about `dataset`:
    - number of representatives in each class
    - message length when separating words with spaces
    
    This allows us to set some hyperparameters of our model, such as the number of tokens it should process at once.
    """
    label_count = {}
    labels = dataset["label"].unique()
    print(len(dataset))
    for label in labels:
        label_count[label] = sum(dataset["label"] == label)
    print(label_count)

inspect_data(dataset)
print(4825 + 747)
