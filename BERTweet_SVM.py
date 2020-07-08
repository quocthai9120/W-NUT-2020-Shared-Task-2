# NOTE: valid.tsv has been renamed to test.tsv. You still need to think about how to "test" the data

# from sklearn.ensemble import SVC

import logging
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from TweetNormalizer import normalizeTweet

from BERT_embeddings import get_bert_embedding
# %matplotlib inline


from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report

# Read data in
df = pd.read_csv(
    '/Users/qthai912/Desktop/VinAI_Intern/W-NUT-2020-Shared-Task-2/train.tsv', sep='\t', lineterminator='\n', header=0)

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
    '/Users/qthai912/Desktop/VinAI_Intern/W-NUT-2020-Shared-Task-2/test.tsv', sep='\t')
X_test = df_test.Text.apply(normalizeTweet)
y_test = df_test.Label

# SVM 'pipeline'
# Vectorizer => Transformer => Classifier
sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                      alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                ])

# Our way
print(type(X_train))
X_train_embeddings = get_bert_embedding(X_train[4373:4375])

# print(len(X_train))
# print(X_train[1106])

# Train SVM
# svm.fit(embedding, label)

#


# sgd.fit(X_train, y_train)

# y_pred = sgd.predict(X_test)

# my_tags = ('INFORMATIVE', 'UNINFORMATIVE')

# print('accuracy %s' % accuracy_score(y_pred, y_test))
# print(classification_report(y_test, y_pred, target_names=my_tags))
