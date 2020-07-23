import torch
import pandas as pd
import numpy as np
from numpy import random
import nltk
from sklearn.metrics import accuracy_score, confusion_matrix
#import matplotlib.pyplot as plt
import re
from transformers import BertTokenizer
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

MAX_LENGTH = 256


# Function to calculate the accuracy of our predictions vs labels


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_f1_score(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(pred_flat, labels_flat)


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


# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model = torch.load("finetune-BERTweet-weights/stage_2_weights.pth",
                   map_location=device)

model.cuda()

# Prepare data to test the model after training
df_test = pd.read_csv('./test.tsv', sep='\t', lineterminator='\n', header=0)
test_text_data = df_test.Text.apply(normalizeTweet)
test_labels = df_test.Label
test_labels = test_labels.replace('INFORMATIVE', 1)
test_labels = test_labels.replace('UNINFORMATIVE', 0)

batch_size = 16
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True)

input_ids_and_att_masks_tuple: Tuple[List, List] = get_input_ids_and_att_masks(
    test_text_data)

prediction_inputs: torch.tensor = torch.cat(
    input_ids_and_att_masks_tuple[0], dim=0)
prediction_masks: torch.tensor = torch.cat(
    input_ids_and_att_masks_tuple[1], dim=0)
prediction_labels: torch.tensor = torch.tensor(test_labels)

# Create the DataLoader.
prediction_data = TensorDataset(
    prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(
    prediction_data, sampler=prediction_sampler, batch_size=batch_size)

################# TEST ##################
total_eval_accuracy = 0

# Prediction on test set
print('Predicting labels for {:,} test sentences...'.format(
    len(prediction_inputs)))
# Put model in evaluation mode
model.eval()
# Tracking variables
predictions, true_labels = [], []
# Predict
count = 0
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids,
                        attention_mask=b_input_mask)
    logits = outputs[0]
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Store predictions and true labels
    for i in range(len(logits)):
        predictions.append(logits[i])
        true_labels.append(label_ids[i])

print("  Accuracy: {0:.4f}".format(
    flat_accuracy(np.asarray(predictions), np.asarray(true_labels))))
print("  F1-Score: {0:.4f}".format(
    get_f1_score(np.asarray(predictions), np.asarray(true_labels))))
