import torch
from global_local_BERTweet_model import BERTweetModelForClassification as global_local_BERTweet
from BERTweetForBinaryClassification import BERTweetForBinaryClassification as original_BERTweet
from newBERTweetModel import newBERTweetModelForClassification as last_four_layers_BERTweet
from BERTweet_all_embeddings_model import BERTweetModelForClassification as all_embeddings_BERTweet

import numpy as np
import argparse

from sklearn.metrics import accuracy_score, f1_score, classification_report

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
from typing import List, Tuple

from torch.utils.data import DataLoader, TensorDataset, SequentialSampler

from TweetNormalizer import normalizeTweet
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import os
import pandas as pd

BATCH_SIZE = 32
MAX_LENGTH = 256


def flat_accuracy(preds, labels) -> np.long:
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_f1_score(preds, labels) -> np.long:
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat)


def get_classification_report(labels, preds):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return classification_report(labels_flat, pred_flat, digits=4)


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


def predict(prediction_dataloader, model, prediction_inputs, device) -> Tuple:
    total_eval_accuracy = 0

    # Prediction on test set
    print('Predicting labels for {:,} test sentences...'.format(
        len(prediction_inputs)))
    # Put model in evaluation mode
    model.to(device)
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
        curr_softmax_outputs: torch.tensor = softmax(torch.tensor(logits))

        # Store predictions, softmax vectors, and true labels
        for i in range(len(logits)):
            predictions.append(logits[i])
            softmax_outputs.append(curr_softmax_outputs[i])
            true_labels.append(label_ids[i])

    return (predictions, true_labels, softmax_outputs)


def vote(predictions_list):
    result = np.zeros(np.array(predictions_list[0]).shape)
    for prd in predictions_list:
        result += np.array(prd)
    result = 1/4 * result
    return result


def main() -> None:
    device: torch.device = setup_device()

    # Prepare data to test the model after training
    df_test = pd.read_csv('./data/test.csv')
    test_text_data = df_test.Text.apply(normalizeTweet)
    test_labels = df_test.Label
    test_labels = test_labels.replace('INFORMATIVE', 1)
    test_labels = test_labels.replace('UNINFORMATIVE', 0)

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
    # prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, batch_size=BATCH_SIZE)

    # Load Models
    original_BERTweet_model = original_BERTweet()
    original_BERTweet_model.load_state_dict(
        torch.load(
            "finetune-BERTweet-weights/stage_2_weights.pth", map_location=device)
    )

    global_local_BERTweet_model = global_local_BERTweet()
    global_local_BERTweet_model.load_state_dict(
        torch.load(
            "global-local-BERTweet-weights/stage_2_weights.pth", map_location=device)
    )

    last_four_layers_BERTweet_model = last_four_layers_BERTweet()
    last_four_layers_BERTweet_model.load_state_dict(
        torch.load(
            "new_finetune-BERTweet-weights/stage_2_weights.pth", map_location=device)
    )

    all_embeddings_BERTweet_model = all_embeddings_BERTweet()
    all_embeddings_BERTweet_model.load_state_dict(
        torch.load(
            "finetune-BERTweet-all-embeddings-weights/stage_2_weights.pth", map_location=device)
    )

    models: List = [
        original_BERTweet_model,
        global_local_BERTweet_model,
        last_four_layers_BERTweet_model,
        all_embeddings_BERTweet_model,
    ]

    predictions_list = []
    true_labels = None
    for model in models:
        predictions, true_labels, softmax_outputs = predict(
            prediction_dataloader, model, prediction_inputs, device)
        predictions_list.append(predictions)

    temp = vote(predictions_list)
    print(temp)
    print(get_classification_report(np.asarray(true_labels), np.asarray(temp)))


if __name__ == "__main__":
    main()