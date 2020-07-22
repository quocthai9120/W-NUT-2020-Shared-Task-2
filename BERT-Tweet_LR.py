# NOTE: valid.tsv has been renamed to test.tsv. You still need to think about how to "test" the data

# from sklearn.ensemble import SVC

import torch
import logging
import pandas as pd
import numpy as np
from numpy import random
# import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from TweetNormalizer import normalizeTweet

from BERT_embeddings import get_bert_embedding
# %matplotlib inline
import os

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report

import pickle

# Read data in
df = pd.read_csv(
    './train.tsv', sep='\t', lineterminator='\n', header=0)

df = df[pd.notnull(df['Label'])]
# print(df.head(10))
# print(df['Text'].apply(lambda x: len(x.split(' '))).sum())

# Normalizing the tweets
df['Text'] = df['Text'].apply(normalizeTweet)

# Prepare data to train the model
X_train = df.Text
y_train = df.Label

# Prepare data to test the model after training
df_test = pd.read_csv(
    './test.tsv', sep='\t')
X_test = df_test.Text.apply(normalizeTweet)
y_test = df_test.Label

X_train_embeddings = None
X_test_embeddings = None
# Train set
train_filename = "./train-embeddings.pkl"
if os.path.isfile('{}'.format(train_filename)):
    infile = open(train_filename, 'rb')
    X_train_embeddings = pickle.load(infile)
    infile.close()
else:
    X_train_embeddings = get_bert_embedding(X_train)
    outfile = open(train_filename, 'wb')
    pickle.dump(X_train_embeddings, outfile)
    outfile.close()

# Test set
test_filename = "./test-embeddings.pkl"
if os.path.isfile('{}'.format(test_filename)):
    infile = open(test_filename, 'rb')
    X_test_embeddings = pickle.load(infile)
    infile.close()
else:
    X_test_embeddings = get_bert_embedding(X_test)
    outfile = open(test_filename, 'wb')
    pickle.dump(X_test_embeddings, outfile)
    outfile.close()


y_train = y_train.replace('INFORMATIVE', 1)
y_train = y_train.replace('UNINFORMATIVE', 0)

y_test = y_test.replace('INFORMATIVE', 1)
y_test = y_test.replace('UNINFORMATIVE', 0)

lr = LogisticRegression()

# Train SVM
lr.fit(X_train_embeddings, y_train)

# Predict on X_test
y_pred = lr.predict(X_test_embeddings)

# Print info on prediction
my_tags = ('INFORMATIVE', 'UNINFORMATIVE')

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred, target_names=my_tags))
