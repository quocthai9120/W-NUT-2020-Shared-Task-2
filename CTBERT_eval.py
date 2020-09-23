import torch
import pandas as pd
import numpy as np
from numpy import random
import nltk
from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
import re
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

from TweetNormalizer import normalizeTweet
from BERT_embeddings import get_bert_embedding
import os
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from typing import Tuple, List
import argparse
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

from CTBERT_model import CTBERTForBinaryClassification


MAX_LENGTH = 256
MODEL_PATH: str = "./BERTweet-covid19-base-cased/"

# Function to calculate the accuracy of our predictions vs labels


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_classification_report(labels, preds):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return classification_report(labels_flat, pred_flat, digits=4)


def get_f1_score(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat)


def get_input_ids_and_att_masks(lines: pd.core.series.Series) -> Tuple[List, List]:
    # Load the CTBERT tokenizer.
    print('Loading CTBERT tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained("CTBERT")

    # Tokenize dataset
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    for sent in lines:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,                      # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=256,
            truncation=True,           # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,   # Construct attn. masks.
            return_tensors='pt',     # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    return tuple([input_ids, attention_masks])


def export_wrong_predictions(preds: np.array, labels: np.array, data: pd.DataFrame) -> None:
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    wrong_pred_index: List = []
    for i in range(len(pred_flat)):
        if pred_flat[i] != labels_flat[i]:
            wrong_pred_index.append(i)
    filtered_data = data[data.index.isin(wrong_pred_index)]
    filtered_data.to_csv('finetune_BERTweet_wrong_preds.csv')


def main() -> None:
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

    # Prepare data to test the model after training
    df_test = pd.read_csv('./data/test.csv')
    test_text_data = df_test.Text.apply(normalizeTweet)
    test_labels = df_test.Label
    test_labels = test_labels.replace('INFORMATIVE', 1)
    test_labels = test_labels.replace('UNINFORMATIVE', 0)

    batch_size = 16

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

    # Load model

    for dir_index in range(5):
        for epoch_index in range(7):
            model = CTBERTForBinaryClassification()
            model.load_state_dict(torch.load(
                "CTBERT-weights-" + str(dir_index) + "/stage_2_weights_epoch_" + str(epoch_index) + ".pth", map_location=device))

            model.cuda()

            ################# TEST ##################
            total_eval_accuracy = 0

            # Prediction on test set
            print('Predicting labels for {:,} test sentences...'.format(
                len(prediction_inputs)))
            # Put model in evaluation mode
            model.eval()
            # Tracking variables
            predictions, softmax_outputs, true_labels = [], [], []
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

                softmax: torch.nn.Softmax = torch.nn.Softmax()
                curr_softmax_outputs: torch.tensor = softmax(
                    torch.tensor(logits))

                # Store predictions, softmax vectors, and true labels
                for i in range(len(logits)):
                    predictions.append(logits[i])
                    softmax_outputs.append(curr_softmax_outputs[i])
                    true_labels.append(label_ids[i])

            print("  Accuracy: {0:.4f}".format(
                flat_accuracy(np.asarray(predictions), np.asarray(true_labels))))
            print("  F1-Score: {0:.4f}".format(
                get_f1_score(np.asarray(predictions), np.asarray(true_labels))))
            print("Report")
            cls_report = get_classification_report(np.asarray(
                true_labels), np.asarray(predictions))
            print(cls_report)
            export_wrong_predictions(np.asarray(predictions),
                                     np.asarray(true_labels), df_test)

            pred_path = "CTBERT-predictions-" + str(dir_index)
            if not os.path.exists(pred_path):
                os.makedirs(pred_path)

            report_file = pred_path + "/preds_epoch_" + \
                str(epoch_index) + "_report.txt"
            f = open(report_file, "w")
            f.write(str(cls_report))
            f.write("\n")
            f.close()

            file = pred_path + "/preds_epoch_" + str(epoch_index) + ".txt"
            f = open(file, "w")

            for i in predictions:
                f.write("{}, {} \n".format(i[0], i[1]))
            f.close()


if __name__ == "__main__":
    main()
