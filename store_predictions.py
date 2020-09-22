import torch
import pandas as pd
import numpy as np

from TweetNormalizer import normalizeTweet
import os

from sklearn.metrics import classification_report

from typing import List, Set
import ensemble_BERTweet


def read_preds_files(input_dir: str) -> List:
    predictions: List = []
    # Enumerate all softmax output files
    for f in os.listdir(input_dir):
        print(f)
        path = os.path.join(input_dir, f)
        arr = np.loadtxt(path, delimiter=',')
        predictions.append(arr)
        print(arr.shape)
    return predictions


def store_df(predictions: List, labels: pd.Series, length_classes: pd.Series) -> pd.DataFrame:
    result: List[List[int]] = []
    for preds in predictions:
        pred_flat = np.argmax(preds, axis=1).flatten()
        result.append(pred_flat)
    df: pd.DataFrame = pd.DataFrame(result).T
    df['gold_label'] = labels.astype('int32')
    df['length_class'] = length_classes.astype('int32')
    return df


def select_models_to_ensemble(df: pd.DataFrame, num_models: int = 7) -> List[int]:
    """
    Return a list of model indexes that can be used for ensemble
    """
    correct_preds_list: List[int] = []
    for i in range(len(df.columns) - 2):
        correct_preds_list.append(len(df[df[i] - df.gold_label == 0]))
    print("Correct Preds List: Out of " + str(len(df)))
    print(correct_preds_list)

    n_best: List[bool] = [True] * len(correct_preds_list)
    for _ in range(len(correct_preds_list) - num_models):
        curr_min: int = correct_preds_list[0]
        curr_min_indx: int = 0
        for i in range(len(correct_preds_list)):
            if correct_preds_list[i] < curr_min:
                curr_min = correct_preds_list[i]
                curr_min_indx = i
        correct_preds_list[curr_min_indx] = 1000000
        n_best[curr_min_indx] = False

    n_best_indx: List[int] = []
    for i in range(len(n_best)):
        if n_best[i]:
            n_best_indx.append(i)

    return n_best_indx


def major_vote(preds_df: pd.DataFrame) -> pd.DataFrame:
    result = np.zeros(len(preds_df.values))
    for column in preds_df.columns:
        if column != 'index':
            result += preds_df[column].values
    result = np.where(result <= (
        (len(preds_df.columns) - 1) / 2), 0, 1)
    preds_df['result'] = pd.Series(result)
    return pd.DataFrame(
        [preds_df['index'], preds_df['result']]).T


def ensemble(predictions_list: pd.DataFrame,
             short_tweets_models: List[int],
             med_tweets_models: List[int],
             long_tweets_models: List[int],
             labels: pd.Series,
             length_classes: pd.Series) -> np.ndarray:
    predictions_df: pd.DataFrame = store_df(
        predictions_list, labels, length_classes)
    short_tweets_preds: pd.DataFrame = predictions_df[predictions_df.length_class ==
                                                      0][short_tweets_models].reset_index()
    short_tweets_major_voting_preds: pd.DataFrame = major_vote(
        short_tweets_preds)

    med_tweets_preds: pd.DataFrame = predictions_df[predictions_df.length_class ==
                                                    1][med_tweets_models].reset_index()
    med_tweets_major_voting_preds: pd.DataFrame = major_vote(
        med_tweets_preds)

    long_tweets_preds: pd.DataFrame = predictions_df[predictions_df.length_class ==
                                                     2][long_tweets_models].reset_index()
    long_tweets_major_voting_preds: pd.DataFrame = major_vote(
        long_tweets_preds)
    major_voting_preds: pd.DataFrame = pd.concat(
        [short_tweets_major_voting_preds, med_tweets_major_voting_preds, long_tweets_major_voting_preds]).sort_values(by='index').set_index('index')
    return major_voting_preds.result.to_numpy()


def main() -> None:
    # Prepare data to test the model after training
    df_test = pd.read_csv('./final_final_data/test.csv')
    test_labels = df_test.Label
    test_tweet_length_class = df_test.Length_class
    test_labels = test_labels.replace('INFORMATIVE', 1)
    test_labels = test_labels.replace('UNINFORMATIVE', 0)

    # Read Predictions
    input_dir = './predictions_original_val'
    predictions: List = read_preds_files(input_dir)
    df: pd.DataFrame = store_df(
        predictions, test_labels, test_tweet_length_class)

    # REMOVE THE LINE AFTER THIS IN REAL PRODUCTION
    # df = df.head(20)

    # Separate classes of tweets
    short_tweets_df: pd.DataFrame = df[df.length_class == 0].reset_index(
        drop=True)
    med_tweets_df: pd.DataFrame = df[df.length_class == 1].reset_index(
        drop=True)
    long_tweets_df: pd.DataFrame = df[df.length_class == 2].reset_index(
        drop=True)

    print("Number of short Tweets: " + str(len(short_tweets_df)))
    print("Number of medium Tweets: " + str(len(med_tweets_df)))
    print("Number of long Tweets: " + str(len(long_tweets_df)))

    short_best_models: List[int] = select_models_to_ensemble(short_tweets_df)
    med_best_models: List[int] = select_models_to_ensemble(med_tweets_df)
    long_best_models: List[int] = select_models_to_ensemble(long_tweets_df)

    print("Short Tweets Best models: " + str(short_best_models))
    print("Med Tweets Best models: " + str(med_best_models))
    print("Long Tweets Best models: " + str(long_best_models))
    major_voting_preds: np.ndarray = ensemble(predictions, short_best_models,
                                              med_best_models, long_best_models, test_labels, test_tweet_length_class)
    print(classification_report(
        test_labels.to_numpy().flatten(), major_voting_preds, digits=4))
    exit()
    df.to_csv('preds.csv', index=False)


if __name__ == "__main__":
    main()
