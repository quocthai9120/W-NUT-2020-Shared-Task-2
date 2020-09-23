import pandas as pd
import numpy as np


def tweet_length(dataframe):
    line_number: int = 0
    for line in dataframe.Text:
        # print("At line " + str(line_number) +
        #       " len is " + str(len(line.split())))
        if len(line.split()) > 100:
            print("Error at line " + str(line_number))
            exit()
        line_number += 1


def main() -> None:
    print("Reading Training Data")
    train: pd.DataFrame = pd.read_csv("./data/train.tsv", sep='\t')
    tweet_length(train)
    train.to_csv(path_or_buf="./data/train.csv", index=False)

    print("Reading Validation Data")
    valid: pd.DataFrame = pd.read_csv("./data/valid.tsv", sep='\t')
    tweet_length(valid)
    valid.to_csv(path_or_buf="./data/valid.csv", index=False)

    print("Reading Testing Data")
    test: pd.DataFrame = pd.read_csv("./data/test.tsv", sep='\t')
    tweet_length(test)
    test.to_csv(path_or_buf="./data/test.csv", index=False)


if __name__ == "__main__":
    main()
