import torch
import pandas as pd
import numpy as np
from TweetNormalizer import normalizeTweet


def main():
    ######################################## Prepare Data ########################################
    # Prepare train data
    df_train: pd.DataFrame = pd.read_csv('./data_join/train.csv')
    df_valid: pd.DataFrame = pd.read_csv('./data_join/test.csv')

    # Normalizing the tweets
    df_train['Text'] = df_train['Text'].apply(normalizeTweet)
    df_valid['Text'] = df_valid['Text'].apply(normalizeTweet)

    # Prepare data to train the model
    train_text_data: pd.core.series.Series = df_train.Text
    train_labels: pd.core.series.Series = df_train.Label.replace(
        {'INFORMATIVE': 1, 'UNINFORMATIVE': 0})

    valid_text_data: pd.core.series.Series = df_valid.Text
    valid_labels: pd.core.series.Series = df_valid.Label.replace(
        {'INFORMATIVE': 1, 'UNINFORMATIVE': 0})

    print(len(train_labels[train_labels == 1]))
    print(len(train_labels[train_labels == 0]))
    print(len(valid_labels[valid_labels == 1]))
    print(len(valid_labels[valid_labels == 0]))


if __name__ == "__main__":
    main()
