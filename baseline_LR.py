# NOTE: valid.tsv has been renamed to test.tsv. You still need to think about how to "test" the data
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
#%matplotlib inline



from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report

# Read data in
df = pd.read_csv('/home/vina/Desktop/W-NUT/train.tsv', sep='\t')
df = df[pd.notnull(df['Label'])]
#print(df.head(10))
#print(df['Text'].apply(lambda x: len(x.split(' '))).sum())

# Normalizing the tweets
df['Text'] = df['Text'].apply(normalizeTweet)
#print(df.tail(10))

# Prepare data to train the model
X_train = df.Text
y_train = df.Label

# Prepare data to test the model after training
df_test = pd.read_csv('/home/vina/Desktop/W-NUT/test.tsv', sep='\t')
X_test = df_test.Text.apply(normalizeTweet)
y_test = df_test.Label

# Logistic regression 'pipeline'
# Vectorizer => Transformer => Classifier
logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

my_tags = ('INFORMATIVE', 'UNINFORMATIVE')

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))