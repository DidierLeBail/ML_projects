import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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
print()

try:
    assert len(dataset) == 5574
except AssertionError as err:
    print(f"nb of samples: {len(dataset.index)}, should be 5574")
    raise err
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
    for label in labels:
        label_count[label] = sum(dataset["label"] == label)
    print("nb of SMS per label:")
    print(label_count)
    print()

    msg_length = []
    for text in dataset["text"]:
        msg_length.append( len(text.split(' ')) )
    print(f"SMS max length: {max(msg_length)}")
    print()
    fig, ax = plt.subplots(1, 1)
    ax.hist(msg_length, bins=20)
    plt.show()

    # separate the length distribution per label
    msg_length = {label: [] for label in labels}
    for label, text in zip(dataset["label"], dataset["text"]):
        msg_length[label].append( len(text.split(' ')) )
    fig, ax = plt.subplots(1, 1)
    for label in labels:
        ax.hist(msg_length[label], bins=20, label=label)
    ax.legend()
    ax.set_title("nb of words per SMS")
    print(f"spam max length: {max(msg_length['spam'])}")
    print()
    plt.show()

# inspect_data(dataset)

def simple_classifier(text: str):
    """Flags a msg as a spam when it contains a number with 11 digits.
    """

def evaluate_classifier(classifier: SpamClassifier, X_test: np.ndarray, Y_test: np.ndarray):
    """Evaluates a spam classifier on a test set by computing the following metrics:
    - nb of true positives (flag a spam as a spam)
    - nb of true negatives (flag a ham as a ham)
    - nb of false positives (flag a ham as a spam)
    - nb of false negatives (flag a spam as a ham)
    """
    labels = ['spam', 'ham']
    mat = np.zeros((2, 2))
    print(mat)
    for text, true_label in zip(X_test, Y_test):
        flag = classifier(text)
        mat[true_label, flag] += 1
    mat /= 5574
    print(mat)

class RandomClassifier(SpamClassifier):
    """A random classifier."""

    def __init__(self, p_spam: float):
        SpamClassifier.__init__(self)
        self.p_spam = p_spam
        self.rng = np.random.default_rng()
    
    def __call__(self, msg: np.ndarray):
        return int(self.rng.random() < self.p_spam)

X_test, Y_test = dataset["text"].to_list(), (dataset["label"] == 'spam').to_list()
Y_test = [int(el) for el in Y_test]
randomClassifier = RandomClassifier(p_spam=0.5)

evaluate_classifier(randomClassifier, X_test, Y_test)
