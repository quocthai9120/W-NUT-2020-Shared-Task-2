import ensemble_BERTweet
import numpy as np
import os

# Hey, add your softmax output directory here
softmax_path = "./export"

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

ensemble_BERTweet.export(
    "average_ensembling_predictions.txt", average_ensembling_predictions)
ensemble_BERTweet.export(
    "major_voting_ensembling_predictions.txt", major_voting_ensembling_predictions)
