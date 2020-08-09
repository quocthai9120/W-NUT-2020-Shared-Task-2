# NOTE: valid.tsv has been renamed to test.tsv. You still need to think about how to "test" the data
import pandas as pd
import numpy as np
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import sys
sys.path.append("/home/vina/W-NUT-2020-Shared-Task-2/")
from TweetNormalizer import normalizeTweet

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report

# Read data in
df = pd.read_csv('./data/train.csv')
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
df_test = pd.read_csv('./data/test.csv')
X_test = df_test.Text.apply(normalizeTweet)
y_test = df_test.Label

#print(X_test.head(20))

# Naive Bayes 'pipeline'
# Vectorizer => Transformer => Classifier
nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

my_tags = ('INFORMATIVE', 'UNINFORMATIVE')

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))