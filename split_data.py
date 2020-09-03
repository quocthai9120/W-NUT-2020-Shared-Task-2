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

# 3 classes: 0-22, 22-44, 44-max => 0, 1, 2, respectively
def classify_tweet_length(dataframe):
    dataframe['Length_class'] = ""
    dataframe.loc[dataframe['Num_words'] < 22, 'Length_class'] = 0
    dataframe.loc[(22 <= dataframe['Num_words']) & (dataframe['Num_words'] <= 44), 'Length_class'] = 1
    dataframe.loc[dataframe['Num_words'] > 44, 'Length_class'] = 2


# Read data in
df_train = pd.read_csv('./data/train.csv')
df_train = df_train[pd.notnull(df_train['Label'])]
<<<<<<< HEAD
print(df_train.shape)

df_valid = pd.read_csv('./data/valid.csv')
print(df_valid.shape)

df_test = pd.read_csv('./data/test.csv')
print(df_test.shape)
=======
tweet_length(df_train)
classify_tweet_length(df_train)
>>>>>>> 4192a7e90f999035ac1b18c4f335d7cd882f2e2b


<<<<<<< HEAD
df_total = np.asarray(df_total)
print(df_total.shape)
print(type(df_total))

X_train, X_test, y_train, y_test = train_test_split(
    df_total[:, 0], df_total[:, 1], train_size=0.9, random_state=RANDOM_STATE)
=======
df_valid = pd.read_csv('./data/valid.csv')
df_valid = df_valid[pd.notnull(df_valid['Label'])]
tweet_length(df_valid)
classify_tweet_length(df_valid)
>>>>>>> 4192a7e90f999035ac1b18c4f335d7cd882f2e2b

df_test = pd.read_csv('./data/test.csv')
df_test = df_test[pd.notnull(df_test['Label'])]
tweet_length(df_test)
classify_tweet_length(df_test)


'''
Write data to new csv files, stored in ./data
'''
df_train.to_csv(path_or_buf="./final_final_data/train.csv", index=False)

<<<<<<< HEAD
testData = {'Text': X_test, 'Label': y_test}
dfTest = pd.DataFrame(data=testData)
dfTest.to_csv(path_or_buf="./data_join/test.csv", index=False)
=======
df_valid.to_csv(path_or_buf="./final_final_data/valid.csv", index=False)

df_test.to_csv(path_or_buf="./final_final_data/test.csv", index=False)


>>>>>>> 4192a7e90f999035ac1b18c4f335d7cd882f2e2b
