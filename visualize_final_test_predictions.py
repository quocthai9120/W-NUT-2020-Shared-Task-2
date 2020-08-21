import pandas as pd
import torch


def count_informative_labels(preds, col):
    count = 0
    for i in range(len(preds)):
        if preds[col].values[i] == 'INFORMATIVE ':
            count += 1
    return count


def main() -> None:
    df_final_test = pd.read_csv('./data/final_test.csv')
    major_voting_predictions = pd.read_csv(
        'major_voting_ensembling_predictions.txt', names=['m'])
    average_ensembling_predictions = pd.read_csv(
        'average_ensembling_predictions.txt', names=['a'])
    df_final_test['major_voting_predictions'] = major_voting_predictions
    df_final_test['average_ensembling_predictions'] = average_ensembling_predictions
    df_final_test.to_csv('Result.csv')
    print("Total INFO class using major voting is " +
          str(count_informative_labels(major_voting_predictions, 'm')))
    print("Total INFO class using avr voting is " +
          str(count_informative_labels(average_ensembling_predictions, 'a')))


if __name__ == "__main__":
    main()
