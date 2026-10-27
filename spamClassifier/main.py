import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from classifier import RandomClassifier, evaluate_classifier
from dataset import SMSdataset

DATADIR = os.path.join(
    os.path.dirname(__file__), "data"
)

def inspect_data(data: SMSdataset):
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

data = SMSdataset(DATADIR)
data.load()
df = data.data

# print(df.head())
# print()

# X_test, Y_test = df["text"].to_list(), df["label"]
# randomClassifier = RandomClassifier(p_spam=0.4)

# evaluate_classifier(randomClassifier, X_test, Y_test)

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2)
vec = TfidfVectorizer(stop_words="english")
X_train_tfidf = vec.fit_transform(X_train)
model = MultinomialNB().fit(X_train_tfidf, y_train)
print(classification_report(y_test, model.predict(vec.transform(X_test))))

# get the most predictive words,
# defined as the mutual info btw the Bernoulli variable associated to the word presence in a msg and the Bernoulli variable associated to the label
"""
2 Bernoulli variables of params p (X) and q (Y)

The mut info btw them writes
I(X, Y) = \sum_{x,y}P(x, y)\log\left(\frac{P(x, y)}{P(x)P(y)}\right) =
= - H(X, Y) - (1 - p)\log(1 - p) - p\log(p) - (1 - q)\log(1 - q) - q\log(q)
"""

