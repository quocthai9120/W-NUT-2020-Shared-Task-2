import torch
import pandas as pd
import numpy as np
from numpy import random
import nltk
from sklearn.metrics import accuracy_score, confusion_matrix

import re
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

from TweetNormalizer import normalizeTweet
from BERT_embeddings import get_bert_embedding
import os
import pickle
from sklearn.metrics import f1_score

from typing import Tuple, List
import argparse
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

from torch import nn

SEED_VAL: int = 912
MAX_LENGTH = 256
BATCH_SIZE = 16


def setup_device() -> torch.device:
    """
    Post: Return torch.device instance repr whether we are using a CUDA GPU or CPU
    """
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
        return torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        return torch.device("cpu")


def get_input_ids_and_att_masks(lines: pd.core.series.Series) -> Tuple[List, List]:
    # Load BPE Tokenizer
    print('Load BPE Tokenizer')
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--bpe-codes',
                        default="./BERTweet_base_transformers/bpe.codes",
                        required=False,
                        type=str,
                        help='path to fastBPE BPE'
                        )
    args: fastBPE = parser.parse_args()
    bpe: argparse.Namespace = fastBPE(args)

    vocab: Dictionary = Dictionary()
    vocab.add_from_file("./BERTweet_base_transformers/dict.txt")

    input_ids: List = []
    attention_masks: List = []
    for line in lines:
        # (1) Tokenize the sentence
        # (2) Add <CLS> token and <SEP> token (<s> and </s>)
        # (3) Map tokens to IDs
        # (4) Pad/Truncate the sentence to `max_length`
        # (5) Create attention masks for [PAD] tokens
        subwords: str = '<s> ' + bpe.encode(line) + ' </s>'  # (1) + (2)
        line_ids: List = vocab.encode_line(
            subwords, append_eos=False, add_if_not_exist=False).long().tolist()  # (3)

        if len(line_ids) < MAX_LENGTH:
            paddings: torch.tensor = torch.ones(
                (1, MAX_LENGTH - len(line_ids)), dtype=torch.long)
            # convert the line_ids to torch tensor
            tensor_line_ids: torch.tensor = torch.cat([torch.tensor(
                [line_ids], dtype=torch.long), paddings], dim=1)
            line_attention_masks: torch.tensor = torch.cat([torch.ones(
                (1, len(line_ids)), dtype=torch.long), torch.zeros(
                (1, MAX_LENGTH - len(line_ids)), dtype=torch.long)], dim=1)
        elif len(line_ids) > MAX_LENGTH:
            tensor_line_ids: torch.tensor = torch.tensor(
                [line_ids[0:MAX_LENGTH]], dtype=torch.long)
            line_attention_masks: torch.tensor = torch.ones(
                (1, MAX_LENGTH), dtype=torch.long)

        input_ids.append(tensor_line_ids)
        attention_masks.append(line_attention_masks)

    return tuple([input_ids, attention_masks])


def main() -> None:
    device: torch.device = setup_device()

    # Load and initialize model
    model = torch.load("finetune-BERTweet-weights/stage_2_weights.pth",
                       map_location=device)
    model.cuda()

    ######################################## Prepare Data ########################################
    # Prepare train data
    print('Preparing training data')
    df_train: pd.DataFrame = pd.read_csv('./train.tsv', sep='\t',
                                         lineterminator='\n', header=0)

    # Normalizing the tweets
    df_train['Text'] = df_train['Text'].apply(normalizeTweet)

    # Prepare data to train the model
    train_text_data: pd.core.series.Series = df_train.Text
    train_labels: pd.core.series.Series = df_train.Label.replace(
        {'INFORMATIVE': 1, 'UNINFORMATIVE': 0})

    print("train_text_data: {}, train_labels: {}".format(
        train_text_data.count(), train_labels.count()))  # 7000, 7000

    ######################################## Tokenization & Input Formatting ########################################
    input_ids_and_att_masks_tuple: Tuple[List, List] = get_input_ids_and_att_masks(
        train_text_data)

    input_ids: torch.tensor = torch.cat(
        input_ids_and_att_masks_tuple[0], dim=0)
    attention_masks: torch.tensor = torch.cat(
        input_ids_and_att_masks_tuple[1], dim=0)
    train_labels: torch.tensor = torch.tensor(train_labels)

    ######################################## Split training and feed to dataloader ########################################
    # Combine the training inputs into a TensorDataset.
    print('Splitting dataset and feed to dataloader')
    dataset: TensorDataset = TensorDataset(
        input_ids, attention_masks, train_labels)

    train_size: int = int(0.9 * len(dataset))
    val_size: int = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader: DataLoader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=BATCH_SIZE  # Trains with this batch size.
    )

    validation_dataloader: DataLoader = DataLoader(
        val_dataset,  # The validation samples.
        # Pull out batches sequentially.
        sampler=SequentialSampler(val_dataset),
        batch_size=BATCH_SIZE  # Evaluate with this batch size.
    )

    ################# get softmax vector #################
    # Put model in evaluation mode
    model.eval()
    # Tracking variables
    softmax_outputs: List = []
    print('Getting softmax vectors')
    for batch in train_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, _ = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask)
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        softmax: nn.Softmax = nn.Softmax()
        curr_softmax_outputs: torch.tensor = softmax(torch.tensor(logits))

        # Store predictions, softmax vectors, and true labels
        for i in range(len(logits)):
            softmax_outputs.append(curr_softmax_outputs[i])


if __name__ == "__main__":
    main()
