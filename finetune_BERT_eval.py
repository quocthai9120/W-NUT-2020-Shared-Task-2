import torch
import pandas as pd
import numpy as np
from numpy import random
import nltk
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import re
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences

from TweetNormalizer import normalizeTweet
from BERT_embeddings import get_bert_embedding
import os
import pickle
from sklearn.metrics import f1_score

# Function to calculate the accuracy of our predictions vs labels


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_f1_score(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(pred_flat, labels_flat)


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

model = torch.load("weights/weights.pth", map_location=device)

# model.cuda()

# Prepare data to test the model after training
df_test = pd.read_csv('./test.tsv', sep='\t', lineterminator='\n', header=0)
test_text_data = df_test.Text.apply(normalizeTweet)
test_labels = df_test.Label
test_labels = test_labels.replace('INFORMATIVE', 1)
test_labels = test_labels.replace('UNINFORMATIVE', 0)

batch_size = 16
MAX_LEN = 256
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True)

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
# For every sentence...
for sent in test_text_data:
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode(
        sent,                      # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    )

    input_ids.append(encoded_sent)
# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN,
                          dtype="long", truncating="post", padding="post")
# Create attention masks
attention_masks = []
# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)
# Convert to tensors.
prediction_inputs = torch.tensor(input_ids)
prediction_masks = torch.tensor(attention_masks)
prediction_labels = torch.tensor(test_labels)

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
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)
    logits = outputs[0]
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Store predictions and true labels
    for i in range(len(logits)):
        predictions.append(logits[i])
        true_labels.append(label_ids[i])

print("  Accuracy: {0:.2f}".format(
    flat_accuracy(np.asarray(predictions), np.asarray(true_labels))))
print("  F1-Score: {0:.2f}".format(
    get_f1_score(np.asarray(predictions), np.asarray(true_labels))))
