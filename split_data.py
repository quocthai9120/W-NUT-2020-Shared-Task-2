'''
Create our own train/validation/test set based on the dataset originally provided by the 
organizers of the competition.
'''

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

RANDOM_STATE: int = 42

# Read data in
df_train = pd.read_csv('./data/train.csv')
df_train = df_train[pd.notnull(df_train['Label'])]
print(df_train.shape)

df_valid = pd.read_csv('./data/valid.csv')
print(df_valid.shape)

df_test = pd.read_csv('./data/test.csv')
print(df_test.shape)

df_total1 = pd.concat([df_train, df_valid], ignore_index=True)
df_total = pd.concat([df_total1, df_test], ignore_index=True)

df_total = np.asarray(df_total)
print(df_total.shape)
print(type(df_total))

X_train, X_test, y_train, y_test = train_test_split(
    df_total[:, 0], df_total[:, 1], train_size=0.9, random_state=RANDOM_STATE)

print("Number of instances in train is {}".format(X_train.size))  # 7200
print("Number of instances in test is {}".format(X_test.size))    # 800


'''
Write data to new csv files, stored in ./data
'''
trainData = {'Text': X_train, 'Label': y_train}
dfTrain = pd.DataFrame(data=trainData)
dfTrain.to_csv(path_or_buf="./data_join/train.csv", index=False)


testData = {'Text': X_test, 'Label': y_test}
dfTest = pd.DataFrame(data=testData)
dfTest.to_csv(path_or_buf="./data_join/test.csv", index=False)
