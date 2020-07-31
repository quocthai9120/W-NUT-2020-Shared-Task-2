'''
Create our own train/validation/test set based on the dataset originally provided by the 
organizers of the competition.
'''

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

RANDOM_STATE: int = 9120

# Read data in
df_train = pd.read_csv('./train.tsv', sep='\t')
df_train = df_train[pd.notnull(df_train['Label'])]

#print("Number of instances in the dataframe is {}".format(df_train['Label'].value_counts()))

df_test = pd.read_csv('./test.tsv', sep='\t')
df_test = df_test[pd.notnull(df_test['Label'])]

#print("Number of instances in the dataframe is {}".format(df_test['Label'].value_counts()))

df_total = pd.concat([df_train, df_test], ignore_index=True)

#print("Number of instances in the dataframe is {}".format(df_total['Label'].value_counts()))

X = df_total['Text']
y = df_total['Label']


'''
Train-val-test split done by calling train_test_split twice 
75:10:15 => Number of instances: (6000:800:1200) 
'''
X_train, X_validAndTest, y_train, y_validAndTest = train_test_split(
    X, y, train_size=0.75, random_state=RANDOM_STATE)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_validAndTest, y_validAndTest, train_size=0.40, random_state=RANDOM_STATE)

print("Number of instances in train is {}".format(X_train.size))  # 6000
print("Number of instances in valid is {}".format(X_valid.size))  # 800
print("Number of instances in test is {}".format(X_test.size))   # 1200


'''
Write data to new csv files, stored in ./data
'''
trainData = {'Text': X_train, 'Label': y_train}
dfTrain = pd.DataFrame(data=trainData)
dfTrain.to_csv(path_or_buf="./data/train.csv", index=False)


validData = {'Text': X_valid, 'Label': y_valid}
dfValid = pd.DataFrame(data=validData)
dfValid.to_csv(path_or_buf="./data/valid.csv", index=False)


testData = {'Text': X_test, 'Label': y_test}
dfTest = pd.DataFrame(data=testData)
dfTest.to_csv(path_or_buf="./data/test.csv", index=False)
