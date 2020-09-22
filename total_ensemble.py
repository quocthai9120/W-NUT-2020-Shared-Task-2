import ensemble_BERTweet
import numpy as np
import os
import pandas as pd
from sklearn.metrics import classification_report
# Hey, add your softmax output directory here
softmax_path = "./predictions_original_val"

listtovote = []

# Enumerate all softmax output files
for f in os.listdir(softmax_path):
    print(f)
    path = os.path.join(softmax_path, f)
    arr = np.loadtxt(path, delimiter=',')
    listtovote.append(arr)
    print(arr.shape)

# Just vote
average_ensembling_predictions = ensemble_BERTweet.average_ensembling(
    listtovote)
major_voting_ensembling_predictions = ensemble_BERTweet.major_voting_ensembling(
    listtovote)

submission_average_file = "./avg.txt"
submission_major_file = "./major.txt"

ensemble_BERTweet.export(submission_average_file,
                         average_ensembling_predictions)
ensemble_BERTweet.export(submission_major_file,
                         major_voting_ensembling_predictions)

print(major_voting_ensembling_predictions)
df_test = pd.read_csv('./final_final_data/test.csv')
test_labels = df_test.Label
test_tweet_length_class = df_test.Length_class
test_labels = test_labels.replace('INFORMATIVE', 1)
test_labels = test_labels.replace('UNINFORMATIVE', 0)

print(classification_report(
    test_labels.to_numpy().flatten(), major_voting_ensembling_predictions, digits=4))
print(classification_report(
    test_labels.to_numpy().flatten(), average_ensembling_predictions, digits=4))
