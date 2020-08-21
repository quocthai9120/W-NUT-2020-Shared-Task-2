import torch
# from global_local_BERTweet_model import BERTweetModelForClassification as global_local_BERTweet
from BERTweetForBinaryClassification import BERTweetForBinaryClassification as original_BERTweet
# from newBERTweetModel import newBERTweetModelForClassification as last_four_layers_BERTweet
# from BERTweet_all_embeddings_model import BERTweetModelForClassification as all_embeddings_BERTweet

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

BATCH_SIZE = 16
MAX_LENGTH = 256

def export(filename, predictions):
    # Write to file here
    f = open(filename)
    for i in range(predictions.size):
        if i==0:
            f.write("UNINFORMATIVE \n")
        else:
            f.write("INFORMATIVE \n")
    f.close()

def flat_accuracy(preds, labels) -> np.long:
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_f1_score(preds, labels) -> np.long:
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat)


def get_classification_report(labels, preds, flattened=False):
    if not flattened:
        pred_flat = np.argmax(preds, axis=1).flatten()
    else:
        pred_flat = preds
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


def average_ensembling(predictions_list):
    num_models: int = len(predictions_list)
    result = np.zeros(np.array(predictions_list[0]).shape)
    for preds in predictions_list:
        result += np.array(preds)
    result = result / num_models
    return result


def major_voting_ensembling(predictions_list):
    num_models: int = len(predictions_list)
    result = np.zeros(np.argmax(predictions_list[0], axis=1).flatten().shape)
    for preds in predictions_list:
        pred_flat = np.argmax(preds, axis=1).flatten()
        result += np.array(pred_flat)
    result = np.where(result <= (num_models / 2), 0, 1)
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
    #original_BERTweet_model = original_BERTweet()
    #original_BERTweet_model.load_state_dict(
    #    torch.load(
    #        "finetune-BERTweet-weights/stage_2_weights.pth/stage_2_weights.pth", map_location=device)
    #)

    BERTweet1 = original_BERTweet()
    BERTweet1.load_state_dict(torch.load("./weights-for-ensembling/BERTweet-1/stage_2_weights.pth", map_location=device))
    

    BERTweet2 = original_BERTweet()
    BERTweet2.load_state_dict(torch.load("./weights-for-ensembling/BERTweet-2/stage_2_weights.pth", map_location=device))


    BERTweet3 = original_BERTweet()
    BERTweet3.load_state_dict(torch.load("./weights-for-ensembling/BERTweet-3/stage_2_weights.pth", map_location=device))


    BERTweet4 = original_BERTweet()
    BERTweet4.load_state_dict(torch.load("./weights-for-ensembling/BERTweet-4/stage_2_weights.pth", map_location=device))


    BERTweet5 = original_BERTweet()
    BERTweet5.load_state_dict(torch.load("./weights-for-ensembling/BERTweet-5/stage_2_weights.pth", map_location=device))


    BERTweet6 = original_BERTweet()
    BERTweet6.load_state_dict(torch.load("./weights-for-ensembling/BERTweet-6/stage_2_weights.pth", map_location=device))


    BERTweet7 = original_BERTweet()
    BERTweet7.load_state_dict(torch.load("./weights-for-ensembling/BERTweet-7/stage_2_weights.pth", map_location=device))


    BERTweet8 = original_BERTweet()
    BERTweet8.load_state_dict(torch.load("./weights-for-ensembling/BERTweet-8/stage_2_weights.pth", map_location=device))


    BERTweet9 = original_BERTweet()
    BERTweet9.load_state_dict(torch.load("./weights-for-ensembling/BERTweet-9/stage_2_weights.pth", map_location=device))


    BERTweet10 = original_BERTweet()
    BERTweet10.load_state_dict(torch.load("./weights-for-ensembling/BERTweet-10/stage_2_weights.pth", map_location=device))


    BERTweet11 = original_BERTweet()
    BERTweet11.load_state_dict(torch.load("./weights-for-ensembling/BERTweet-11/stage_2_weights.pth", map_location=device))


    BERTweet12 = original_BERTweet()
    BERTweet12.load_state_dict(torch.load("./weights-for-ensembling/BERTweet-12/stage_2_weights.pth", map_location=device))


    BERTweet13 = original_BERTweet()
    BERTweet13.load_state_dict(torch.load("./weights-for-ensembling/BERTweet-13/stage_2_weights.pth", map_location=device))


    BERTweet14 = original_BERTweet()
    BERTweet14.load_state_dict(torch.load("./weights-for-ensembling/BERTweet-14/stage_2_weights.pth", map_location=device))

    models: List = [
        #original_BERTweet_model,
        #global_local_BERTweet_model,
        #last_four_layers_BERTweet_model,
        #all_embeddings_BERTweet_model,
        BERTweet1, BERTweet2, BERTweet3, BERTweet4, BERTweet5, BERTweet6, BERTweet7, BERTweet8, BERTweet9, BERTweet10, BERTweet11, BERTweet12, BERTweet13
    ]



    # global_local_BERTweet_model = global_local_BERTweet()
    # global_local_BERTweet_model.load_state_dict(
    #     torch.load(
    #         "global-local-BERTweet-weights/stage_2_weights.pth/stage_2_weights.pth", map_location=device)
    # )

    # last_four_layers_BERTweet_model = last_four_layers_BERTweet()
    # last_four_layers_BERTweet_model.load_state_dict(
    #     torch.load(
    #         "new_finetune-BERTweet-weights/stage_2_weights.pth/stage_2_weights.pth", map_location=device)
    # )

    # all_embeddings_BERTweet_model = all_embeddings_BERTweet()
    # all_embeddings_BERTweet_model.load_state_dict(
    #     torch.load(
    #         "finetune-BERTweet-all-embeddings-weights/stage_2_weights.pth/stage_2_weights.pth", map_location=device)
    # )



    predictions_list = []
    true_labels = None
    for model in models:
        predictions, true_labels, softmax_outputs = predict(
            prediction_dataloader, model, prediction_inputs, device)
        predictions_list.append(predictions)

    average_ensembling_predictions = average_ensembling(predictions_list)
    print(get_classification_report(np.asarray(true_labels),
                                    np.asarray(average_ensembling_predictions)))
    major_voting_ensembling_predictions = major_voting_ensembling(
        predictions_list)
    print(get_classification_report(np.asarray(true_labels),
                                    np.asarray(major_voting_ensembling_predictions), flattened=True))
    

    fileavg = "./avg-predictions.txt"
    filemajor = "./major-predictions.txt" 

    export(fileavg, average_ensembling_predictions)
    export(filemajor, major_voting_ensembling_predictions)



if __name__ == "__main__":
    main()
