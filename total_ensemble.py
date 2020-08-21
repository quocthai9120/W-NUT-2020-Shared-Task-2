import ensemble_BERTweet
import numpy as np
import os

# Hey, add your softmax output directory here
softmax_path = "/home/ubuntu/W-NUT-2020-Shared-Task-2/export"

listtovote = []
# Enumerate all softmax output files
for f in os.listdir(softmax_path):
    path = os.path.join(softmax_path, f)
    arr = np.loadtxt(path, delimiter=',')
    listtovote.append(arr)
    print(arr.shape)

# Just vote
ensemble_BERTweet.average_ensembling(listtovote)
ensemble_BERTweet.major_voting_ensembling(listtovote)
