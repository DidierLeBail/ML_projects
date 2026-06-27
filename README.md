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
