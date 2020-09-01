'''
Create our own train/validation/test set based on the dataset originally provided by the 
organizers of the competition.
'''

import pandas as pd
import numpy as np

def tweet_length(dataframe):
    word_arr = []
    for line in dataframe.Text:
        word_arr.append(len(line.split()))
    dataframe['Num_words'] = word_arr


# Read data in
df_train = pd.read_csv('./data/train.csv')
df_train = df_train[pd.notnull(df_train['Label'])]
tweet_length(df_train)


df_valid = pd.read_csv('./data/valid.csv')
df_valid = df_valid[pd.notnull(df_valid['Label'])]
tweet_length(df_valid)

df_test = pd.read_csv('./data/test.csv')
df_test = df_test[pd.notnull(df_test['Label'])]
tweet_length(df_test)


'''
Write data to new csv files, stored in ./data
'''
df_train.to_csv(path_or_buf="./final_final_data/train.csv", index=False)

df_valid.to_csv(path_or_buf="./final_final_data/valid.csv", index=False)

df_test.to_csv(path_or_buf="./final_final_data/test.csv", index=False)
