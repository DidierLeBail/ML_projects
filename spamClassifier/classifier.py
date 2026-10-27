"""Defines several binary classifiers for the SMS dataset.
"""

__all__ = [
    "SpamClassifier", "evaluate_classifier", "RandomClassifier"
]

import numpy as np

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

def evaluate_classifier(classifier: SpamClassifier, X_test: np.ndarray, Y_test: np.ndarray):
    """Evaluates a spam classifier on a test set by computing the following metrics:
    - nb of true positives (flag a spam as a spam)
    - nb of true negatives (flag a ham as a ham)
    - nb of false positives (flag a ham as a spam)
    - nb of false negatives (flag a spam as a ham)
    """
    labels = ['spam', 'ham']
    mat = np.zeros((2, 2))
    for text, true_label in zip(X_test, Y_test):
        flag = classifier(text)
        mat[true_label, flag] += 1
    for row in mat:
        row /= np.sum(row)
    print(mat)

class RandomClassifier(SpamClassifier):
    """A random classifier."""

    def __init__(self, p_spam: float):
        SpamClassifier.__init__(self)
        self.p_spam = p_spam
        self.rng = np.random.default_rng()
    
    def __call__(self, msg: np.ndarray):
        return int(self.rng.random() < self.p_spam)

def simple_classifier(text: str):
    """Flags a msg as a spam when it contains a number with 11 digits.
    """
