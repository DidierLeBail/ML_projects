"""We use a multinomial Naive Bayes and a logistic model to separate spams from hams in SMS.

A tf-idf vectorizer is used for tokenization of the raw messages.
We experiment both with unigrams, and both unigrams and bigrams.
For each experiment, we display the most predictive words and the classifier performance.
"""

from typing import Literal
import os
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.special import log_softmax

from dataset import loadSMSdataset

DATADIR = os.path.join(
    os.path.dirname(__file__), "data"
)

def inspect_data(data: pd.DataFrame):
    """Computes some basic statistics about `dataset`:
    - number of representatives in each class
    - message length when separating words with spaces
    """
    dataset = data.data
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

def getPredWordsMultiNB(vocab: dict[int, str], mod: MultinomialNB):
    mostPredWordsSpam = []
    mostPredWordsHam = []

    params = mod.feature_log_prob_
    for word, idx_word in vocab.items():
        mostPredWordsSpam.append((word, params[1, idx_word]))
        mostPredWordsHam.append((word, params[0, idx_word]))

    mostPredWordsSpam.sort(key=lambda el: el[1], reverse=True)
    mostPredWordsHam.sort(key=lambda el: el[1], reverse=True)
    return mostPredWordsSpam, mostPredWordsHam

def getPredWordsLogistic(vocab: dict[int, str], mod: LogisticRegression):
    params = log_softmax(
        np.concatenate([mod.coef_, np.ones(np.shape(mod.coef_))], axis=0),
        axis=0
    )
    
    mostPredWordsSpam = []
    mostPredWordsHam = []

    for word, idx_word in vocab.items():
        mostPredWordsSpam.append((word, params[0, idx_word]))
        mostPredWordsHam.append((word, params[1, idx_word]))

    mostPredWordsSpam.sort(key=lambda el: el[1], reverse=True)
    mostPredWordsHam.sort(key=lambda el: el[1], reverse=True)
    return mostPredWordsSpam, mostPredWordsHam

def xp(
    unigrams: bool,
    model: Literal['logistic', 'multiNB']
):
    """Display the classifier performance and most predictive words
    on the SMS dataset for the chosen settings.
    """
    match unigrams:
        case True:
            ngram_range = (1, 1)
        case _:
            ngram_range = (1, 2)

    match model:
        case 'logistic':
            modClass = LogisticRegression()
            getPredWords = getPredWordsLogistic
        case 'multiNB':
            modClass = MultinomialNB()
            getPredWords = getPredWordsMultiNB
        case _:
            raise ValueError("check the value of the arg 'model'")
    
    df = loadSMSdataset(DATADIR)

    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2)
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range
    )
    X_train_tfidf = vec.fit_transform(X_train)

    vocab = vec.vocabulary_
    print(f"vocabulary size: {len(vocab)}")
    print()
    print("some examples of words:")
    for _, key in zip(range(5), vocab):
        print(f"\t{key}")
    print()

    mod = modClass.fit(X_train_tfidf, y_train)
    print("performance report of the classifier:")
    print(classification_report(y_test, mod.predict(vec.transform(X_test))))
    print()

    # identify the most predictive words: how much is a word predictive of a given class c
    # is defined here by its weight associated to belonging to c in the log-space (where we get a linear classifier)
    # we visualize them by plotting this weight wrt word, with the words sorted by decreasing weight
    # an element of mostPredWord<Class> is a tuple (word, weight for <Class>)
    mostPredWordsSpam, mostPredWordsHam = getPredWords(vocab, mod)

    # the nb of words to display
    n_words = 10
    fontsize = 16
    _, ax = plt.subplots(1, 1)
    ax.set_title(f"{n_words} most predictive words for a {model} classifier", fontsize=fontsize)
    ax.set_ylabel("log-proba of word given class", fontsize=fontsize)
    
    X, heights = zip(*mostPredWordsSpam[:n_words])
    ax.bar(
        X, height=heights, label='spam'
    )
    
    X, heights = zip(*mostPredWordsHam[:n_words])
    heights = [-h for h in heights]
    ax.bar(
        X, height=heights, label='ham'
    )
    ax.legend(fontsize=fontsize)

xp(
    unigrams=True,
    model='multiNB'
)

xp(
    unigrams=True,
    model='logistic'
)

plt.show()
