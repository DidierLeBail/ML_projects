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
From the confusion matrix, we can derive the model accuracy, or the probability that it classifies a sample correctly.
It writes:
```math
A = \sum_{k}p_{k}C_{k,k}
```
where $p_k$ denotes the proportion of samples belonging to the class $k$.

### Baseline: random classifier
As a starting point, we consider a random classifier.
This is a stochastic model characterized by a single trainable parameter, the probability $p$ of flagging a message as a spam.

What is the best performance possible of a random classifier?
Considering that the true proportion of spam messages is $p_g$ ($g$ standing for ground truth), let us derive its confusion matrix:

$C_{k,l}=$ Proba that the model returns the class $l$ given the message belongs to class $k$
= Proba that the model returns the class $l$

So we have:
$$
C =
\begin{bmatrix}
    1 - p & p \\
    1 - p & p
\end{bmatrix}
$$

Its accuracy writes:
$$
A(p)=(1-p_g) (1-p) + p_g p = 1 - p_g + (2 p_g - 1) p
$$
which is maximized for
$$
p=H\left(
p_g-\frac{1}{2}
\right)
$$
where $H$ denotes the Heaviside function.

In any case, the maximum accuracy of a random classifier is
$$
A_{\text{max}}=\max(p_g,1-p_g)
$$

### Best classifier
Let us denote by $f(T)$ the probability that the classifier flags the message $T$ as a spam.
The best performance would occur if the $f(T)$ could all be tuned independently from each other.
We have that:

$$
\begin{align}
A[f]
&=
\sum_{T}P(\text{spam},T)f(T)+\sum_{T}P(\text{ham},T)(1-f(T))
\\
&=
P(\text{ham})+\sum_{T}f(T)\left(P(\text{spam},T)-P(\text{ham},T)\right)
\end{align}
$$
Since the $f(T)$ are independent trainable parameters, the maximum accuracy is reached for
$$
f(T)=H\left(
P(\text{spam},T)-P(\text{ham},T)
\right)=
H\left(
P(\text{spam}|T)-\frac{1}{2}
\right)
$$

The corresponding confusion matrix writes:
$$
\begin{cases}
C_{0,0}=\sum_{T}P(T|\text{ham})H\left(P(\text{ham}|T)-\frac{1}{2}\right)
\\
C_{1,1}=\sum_{T}P(T|\text{spam})H\left(P(\text{spam}|T)-\frac{1}{2}\right)
\end{cases}
$$

Of course, this classifier cannot be implemented in practice, since we do not know the value of $H(P\left(\text{spam}|T)-\frac{1}{2}\right)$ for every message $T$.

### Multinomial Bayes classifier (MBC)
Multinomial Bayes is a common NLP technique used to classify messages, which is equivalent to unigram language models.
It can be trained in linear size of the training set and inference time is linear in the size of the messages.

Given a sequence of tokens $T=(t_1, \cdots, t_n)$, the probability that the NBC flags the sequence as a spam writes:
$$
P^{\text{NBC}}(\text{flag}(T)=\text{spam})=
\frac{P^{\text{NBC}}(\text{spam})P^{\text{NBC}}(T|\text{spam})}{P^{\text{NBC}}(\text{spam})P^{\text{NBC}}(T|\text{spam})+P^{\text{NBC}}(\text{ham})P^{\text{NBC}}(T|\text{ham})}
$$
where
$$
P^{\text{NBC}}(T|\text{spam})=\prod_{i=1}^{n}P^{\text{NBC}}(t_{i}|\text{spam})
$$

What is the number of independent trainable parameters of a NBC?
Let us introduce $V$ the vocabulary, or set of possible tokens that compose a message.

Let us introduce the following notations:
$$
\begin{cases}
\forall t\in V,
p_{t}=P^{\text{NBC}}(t|\text{spam}),q_{t}=P^{\text{NBC}}(t|\text{ham})
\\
p=P^{\text{NBC}}(\text{spam})
\\
q=P^{\text{NBC}}(\text{ham})=1-p
\\
\forall \text{ message }T,\forall t\in V,n^{T}_{t}=\text{nb of occurrences of $t$ in $T$}
\\
\forall\vec{n}\in\mathbb{N}^{V},
f(\vec{n})=P^{\text{NBC}}(\text{flag}(T)=\text{spam}),
\text{where }\vec{n}^{T}=\vec{n}
\end{cases}
$$

Then, we have that:
$$
\begin{align}
\forall\vec{n}\in\mathbb{N}^{V},
f(\vec{n})
&=
\frac{p\prod_{t\in V}p_{t}^{n_{t}}}{p\prod_{t\in V}p_{t}^{n_{t}}+q\prod_{t\in V}q_{t}^{n_{t}}}
\\
&=
\frac{1}{1+\exp\left(\alpha+\vec{\alpha}\cdot\vec{n}\right)}
\end{align}
$$
where we have introduced:
$$
\begin{cases}
\alpha=\log\left(\frac{q}{p}\right)
\\
\forall t\in V,\alpha_{t}=\log\left(\frac{q_t}{p_t}\right)
\end{cases}
$$

The $\alpha$ and the $\alpha_{t}$ are exactly the independent trainable parameters of the NBC, which makes $1+|V|$ trainable parameters.

How to determine the parameters maximizing the NBC accuracy?
Denoting by $\theta=(\alpha,\vec{\alpha})$ the model parameters, the accuracy writes:
$$
\begin{align*}
A[\theta]
&=
\sum_{\vec{n}\in\mathbb{N}^{V}}
P(\text{spam},\vec{n})f(\vec{n},\theta)+\sum_{\vec{n}\in\mathbb{N}^{V}}P(\text{ham},\vec{n})(1-f(\vec{n},\theta))
\\
&=
P(\text{ham})+
\sum_{\vec{n}\in\mathbb{N}^{V}}
f(\vec{n},\theta)(P(\text{spam},\vec{n})-P(\text{ham},\vec{n}))
\end{align*}
$$

Note that here:
$$
P(\text{spam},\vec{n})=
\sum_{T|\vec{n}^{T}=\vec{n}}P(\text{spam},T)
$$

Since the NBC output is supposed to match $P(\text{spam}|\vec{n})$ when the messages are generated according to a unigram distribution, we expect in that case the optimal $\theta$ to satisfy:
$$
f(\vec{n},\theta)=
H\left(P(\text{spam}|\vec{n})-\frac{1}{2}\right)
$$

However, given that $f$ is a sigmoid, it cannot fit a Heaviside function.
To enhance the expressivity of our model, let us then consider that a message will be flagged as a spam with probability 1 as soon as the NBC estimates that it is a spam with a probability greater than 1/2.
Said otherwise, we are now considering $H(f-\frac{1}{2})$ instead of $f$.
We have that:
$$
\begin{align*}
H\left(f(\vec{n},\theta)-\frac{1}{2}\right)
&=
H\left(\frac{1}{1+e^{\alpha+\vec{\alpha}\cdot\vec{n}}}-\frac{1}{2}\right)
\\
&=
1-H\left(\alpha+\vec{\alpha}\cdot\vec{n}\right)
\end{align*}
$$

This changes the accuracy:
$$
\begin{align*}
A[\theta]
&=
P(\text{ham})+
\sum_{\vec{n}\in\mathbb{N}^{V}}
(1-H\left(\alpha+\vec{\alpha}\cdot\vec{n}\right))(P(\text{spam},\vec{n})-P(\text{ham},\vec{n}))
\\
&=
P(\text{spam})-
\sum_{\vec{n}\in\mathbb{N}^{V}}
H\left(\alpha+\vec{\alpha}\cdot\vec{n}\right)
(P(\text{spam},\vec{n})-P(\text{ham},\vec{n}))
\end{align*}
$$

Let us rewrite the terms to make $P(\text{spam}|\vec{n})$ explicit.
We have:
$$
\begin{align*}
P(\text{spam},\vec{n})-P(\text{ham},\vec{n})
&=
P(\vec{n})
\frac{P(\text{spam},\vec{n})-P(\text{ham},\vec{n})}{P(\text{spam},\vec{n})+P(\text{ham},\vec{n})}
\\
&=
P(\vec{n})
\left[2
\frac{P(\text{spam},\vec{n})}{P(\text{spam},\vec{n})+P(\text{ham},\vec{n})}
-1
\right]
\\
&=
2P(\vec{n})\left[P(\text{spam}|\vec{n})-\frac{1}{2}\right]
\end{align*}
$$

We deduce that:
$$
A[\theta]
=
P(\text{spam})-2
\sum_{\vec{n}\in\mathbb{N}^{V}}
P(\vec{n})
\left(P(\text{spam}|\vec{n})-\frac{1}{2}\right)
H\left(\alpha+\vec{\alpha}\cdot\vec{n}\right)
$$

In order to maximize its accuracy, a NBC should ideally have $\alpha+\vec{\alpha}\cdot\vec{n}$ and $P(\text{spam}|\vec{n})-\frac{1}{2}$ of opposite signs, which corresponds to $P(\text{spam}|\vec{n})-\frac{1}{2}$ and $P^{\text{NBC}}(\text{spam}|\vec{n})-\frac{1}{2}$ having the same signs.
However, in the case the messages are *not* generated according to a unigram distribution, this will not be possible for every $\vec{n}$.

In practice, the strategy used to train a NBC is to sample the unigram probabilities from the dataset and use them as the NBC parameters.
By using a simple counter-example, we will show here that the accuracy is not maximized by such a choice.
To do this, let us introduce the notations:
$$
\begin{cases}
\lambda_{\vec{n}}=P(\vec{n})
\left(P(\text{spam}|\vec{n})-\frac{1}{2}\right)
\\
p_g=P(\text{spam})
\\
\alpha_{g}=\log\left(\frac{1-p_g}{p_g}\right)
\end{cases}
$$

Then, let us consider a simple case where $|V|=1$ and $||\vec{n}||_{1}\leq1$.
For that toy data distribution, we have:
$$
A[\theta]
=
P(\text{spam})-2\left(
\lambda_{0}H(\alpha)+\lambda_{1}H(\alpha+\alpha_{1})
\right)
$$

Here, maximizing the accuracy is simple and summarized by the following table, depending on the signs of $\lambda_{0}$ and $\lambda_{1}$:
| $\lambda_{0}$ | $\lambda_{1}$ | $\alpha$ | $\alpha_{1}$ |
| --- | --- | --- | --- |
| $>0$ | $>0$ | $<0$ | $<-\alpha$ |
| $>0$ | $<0$ | $<0$ | $>-\alpha$ |
| $<0$ | $>0$ | $>0$ | $<-\alpha$ |
| $<0$ | $<0$ | $>0$ | $>-\alpha$ |

Is that compatible with the data statistics, i.e. the NBC parameters returned by a standard training algorithm?

If that is the case, we would need in particular that $\lambda_{0}>0$ is always compatible with $\alpha_{g}<0$, or equivalently that
$P(\text{spam}|0)>\frac{1}{2}$ is compatible with $p_g>\frac{1}{2}$.
Let us exhibit a data distribution for which this is not the case.
Let us introduce $\epsilon>0$ such that:
$$
\begin{cases}
P(\text{spam},0)=\left(\frac{1}{2}+\epsilon\right)P(0)
\\
P(\text{spam},1)=\frac{1}{2}-\epsilon-\left(\frac{1}{2}+\epsilon\right)P(0)
\\
P(\text{ham},0)=\left(\frac{1}{2}-\epsilon\right)P(0)
\\
P(\text{ham},1)=\frac{1}{2}+\epsilon-\left(\frac{1}{2}-\epsilon\right)P(0)
\end{cases}
$$

When this distribution is legitimate, it automatically satisfies:
$$
\begin{cases}
P(\text{spam}|0)=\frac{1}{2}+\epsilon
\\
p_g=\frac{1}{2}-\epsilon
\end{cases}
$$
which would yield us the counter-example we seek.

Now, when $\epsilon\xrightarrow{}0$, it can be shown (cf expression above) that any value between 0 and 1 for $P(0)$ would make this distribution legitimate.
Then, we are assured of the existence of a couple $(\epsilon,P(0))$ that makes the joint distribution above legitimate.

As a conclusion, depending on the particular data distribution, the classifier accuracy is *not* maximized when the classifier distribution is built from the data statistics, i.e. we should generally have that:
$$
\frac{P^{\text{NBC}}(t|\text{spam})}{P^{\text{NBC}}(t|\text{ham})}
\neq
\frac{P(t|\text{spam})}{P(t|\text{ham})}
$$

However, when the data are generated according to a unigram distribution, as assumed in the NBC, the equality is recovered.
