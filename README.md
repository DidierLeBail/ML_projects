# ML_projects
To show employers I have the ML skills they are looking for.

This repo implements some of the Machine Learning projects described [here](https://www.dataquest.io/blog/machine-learning-projects-for-beginners-to-advanced/).

# Beginner project: spam classifier
The goal here is to detect automatically an SMS as being a spam or not.
We extend the project by adding a third category, corresponding to the model being unsure whether the SMS is a spam.
These SMS are the only ones that should be inspected manually.

## Dataset
The SMS dataset can be downloaded [here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
It does not have any missing values.
After unzipping, the `.txt` file `SMSpamCollection` was moved to the `data` directory.

## The goal
The goal of this repo is twofold:
- implement simple NLP techniques like multinomial Bayes classification
- understand the formal limitations of these techniques
- implement a non-statistical technique, where effort is put on how to define a spam

### Generic workflow
A classifier is a model taking as input a vector-valued time series of fixed length, and returns the probability of flagging this time series as a spam.
The evaluation of a classifier is done by computing its confusion matrix on the test set.
This matrix is 2 x 2 here, with the entries defined as:

$C_{k,l}=$ proportion of messages from class $k$ considered to be in class $l$

For instance, $C_{0,1}=$ proportion of hams flagged as spams.

PASS

### Baseline: random classifier
As a starting point, we consider a random classifier.
This is a stochastic model characterized by a single trainable parameter, the probability $p$ of flagging a message as a spam.

What is the best performance possible of a random classifier?
Considering that the true proportion of spam messages is $p_g$ ($g$ standing for ground truth), let us derive its confusion matrix.

