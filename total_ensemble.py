import ensemble_BERTweet
import numpy as np
import os

# Hey, add your softmax output directory here
softmax_path = "./export"

short_listtovote = ['a', 'b', 'c']
med_l... = ['a', 'd', 'e']
long_ = [....]

# Enumerate all softmax output files
for f in os.listdir(softmax_path):
    path = os.path.join(softmax_path, f)
    arr = np.loadtxt(path, delimiter=',')
    listtovote.append(arr)
    print(arr.shape)

# Just vote
avg = ensemble_BERTweet.average_ensembling(listtovote)
major = ensemble_BERTweet.major_voting_ensembling(listtovote)

submission_average_file = "./submission-avg.txt"
submission_major_file = "./submission-major.txt"

ensemble_BERTweet.export(submission_average_file, avg)
ensemble_BERTweet.export(submission_major_file, major)


def classify_length():

# pre: dataframe
# 0-22, 22-44, 44-max 
# post: add a new column to the dataframe (0 - 1 - 2)