# NOTE: valid.tsv has been renamed to test.tsv. You still need to think about how to "test" the data
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import re
sys.path.append("/home/vina/W-NUT-2020-Shared-Task-2/")
from TweetNormalizer import normalizeTweet

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
import torch


# Read data in
df = pd.read_csv('data/train.csv')
df = df[pd.notnull(df['Label'])]

# Count number of instances to read data in
# Quotation error
print("Number of instances in the dataframe is {}".format(df['Label'].value_counts()))


#print(df.head(10))
#print(df['Text'].apply(lambda x: len(x.split(' '))).sum())

# Normalizing the tweets
df['Text'] = df['Text'].apply(normalizeTweet)
#print(df.tail(10))

# Prepare data to train the model
X_train = df.Text
y_train = df.Label

# Prepare data to test the model after training
df_test = pd.read_csv('data/test.csv')
X_test = df_test.Text.apply(normalizeTweet)
y_test = df_test.Label

# Logistic regression 'pipeline'
# Vectorizer => Transformer => Classifier
logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5, max_iter=1000)),
               ])

logreg.fit(X_train, y_train)

y_pred = logreg.predict_proba(X_test)

list_ypred = []

for i in y_pred:
    tensor_i = torch.from_numpy(i)
    list_ypred.append(tensor_i)

print(list_ypred)

torch.save(list_ypred, "softmax/lr_softmax/test_softmax.pt")

#my_tags = ('INFORMATIVE', 'UNINFORMATIVE')

#print('accuracy %s' % accuracy_score(y_pred, y_test))
#print(classification_report(y_test, y_pred, target_names=my_tags))