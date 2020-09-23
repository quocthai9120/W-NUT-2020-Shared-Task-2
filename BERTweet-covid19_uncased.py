import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, random_split
import argparse
import pandas as pd
import numpy as np
import random
import datetime
import time
from TweetNormalizer import normalizeTweet
from transformers import AdamW, get_linear_schedule_with_warmup
from BERTweet_covid19_uncased_model import BERTweetForBinaryClassification

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

from typing import List, Tuple
from sklearn.metrics import f1_score
import os


MAX_LENGTH: int = 256
# SEED_VAL: int = 69
BATCH_SIZE: int = 32
MODEL_PATH: str = "./BERTweet-covid19-base-uncased/"
# MODEL_WEIGHTS_PATH: str = "./BERTweet-covid19-uncased-weights-"


def format_time(elapsed) -> str:
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def flat_accuracy(preds, labels) -> np.long:
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_f1_score(preds, labels) -> np.long:
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat)


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
                        default=MODEL_PATH + "bpe.codes",
                        required=False,
                        type=str,
                        help='path to fastBPE BPE'
                        )
    args: fastBPE = parser.parse_args()
    bpe: argparse.Namespace = fastBPE(args)

    vocab: Dictionary = Dictionary()
    vocab.add_from_file(MODEL_PATH + "vocab.txt")

    input_ids: List = []
    attention_masks: List = []
    for line in lines:
        # (1) Tokenize the sentence
        # (2) Add <CLS> token and <SEP> token (<s> and </s>)
        # (3) Map tokens to IDs
        # (4) Pad/Truncate the sentence to `max_length`
        # (5) Create attention masks for [PAD] tokens
        subwords: str = '<s> ' + \
            bpe.encode(line.lower()) + ' </s>'  # (1) + (2)
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


def save_model_weights(model, file_name: str, model_path) -> None:
    # Save model weights
    model_weights = model_path

    # Create output directory if needed
    if not os.path.exists(model_weights):
        os.makedirs(model_weights)

    print("Saving model to %s" % (model_weights + file_name))
    torch.save(model.state_dict(), model_weights + file_name)


def stage_1_training(model, train_dataloader, validation_dataloader, device, EPOCHS, MODEL_WEIGHTS_PATH, SEED_VAL) -> None:
    ######################################## Freeze BERTweet for stage 1 training ########################################
    for _, param in model.named_parameters():
        param.requires_grad = False

    model.classifier.weight.requires_grad = True
    model.classifier.bias.requires_grad = True
    model.dense.weight.requires_grad = True
    model.dense.bias.requires_grad = True
    model.dense_2.weight.requires_grad = True
    model.dense_2.bias.requires_grad = True

    # Tell pytorch to run this model on the GPU.
    if device == torch.device("cuda"):
        model.cuda()

    ######################################## Setup Optimizer ########################################
    optimizer = AdamW(model.parameters(),
                      lr=10e-5,  # args.learning_rate - default is 5e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    EPOCHS: int = EPOCHS
    total_steps: int = len(train_dataloader) * EPOCHS

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    ######################################## Training ########################################
    random.seed(SEED_VAL)
    np.random.seed(SEED_VAL)
    torch.manual_seed(SEED_VAL)
    torch.cuda.manual_seed_all(SEED_VAL)

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats: List = []

    # For each epoch...
    for epoch_i in range(0, EPOCHS):
        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            loss, logits = model(b_input_ids,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.4f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        total_eval_f1 = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                (loss, logits) = model(b_input_ids,
                                       attention_mask=b_input_mask,
                                       labels=b_labels)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            total_eval_f1 += get_f1_score(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

        avg_val_f1 = total_eval_f1 / len(validation_dataloader)
        print("  F1: {0:.4f}".format(avg_val_f1))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.4f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        # Save weights
        save_model_weights(model, "/stage_1_weights.pth", MODEL_WEIGHTS_PATH)


def stage_2_training(model, train_dataloader, validation_dataloader, device, EPOCHS, MODEL_WEIGHTS_PATH, SEED_VAL) -> None:
    ######################################## Unfreeze BERTweet for stage 2 training ########################################
    for _, param in model.named_parameters():
        param.requires_grad = True

    # Tell pytorch to run this model on the GPU.
    model.cuda()

    ######################################## Setup Optimizer ########################################
    optimizer = AdamW(model.parameters(),
                      lr=3e-5,  # args.learning_rate - default is 5e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    EPOCHS = EPOCHS
    total_steps = len(train_dataloader) * EPOCHS

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    ######################################## Training ########################################
    random.seed(SEED_VAL)
    np.random.seed(SEED_VAL)
    torch.manual_seed(SEED_VAL)
    torch.cuda.manual_seed_all(SEED_VAL)

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats: List = []

    # For each epoch...
    for epoch_i in range(0, EPOCHS):
        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            loss, logits = model(b_input_ids,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.4f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        total_eval_f1 = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                (loss, logits) = model(b_input_ids,
                                       attention_mask=b_input_mask,
                                       labels=b_labels)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)
            total_eval_f1 += get_f1_score(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.4f}".format(avg_val_accuracy))

        avg_val_f1 = total_eval_f1 / len(validation_dataloader)
        print("  F1: {0:.4f}".format(avg_val_f1))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.4f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        # Save weights
        save_model_weights(
            model, "/stage_2_weights_epoch_" + str(epoch_i) + ".pth", MODEL_WEIGHTS_PATH)

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(
        format_time(time.time()-t0)))


def main(SEED_VAL, MODEL_WEIGHTS_PATH):
    print("Current Seed is " + str(SEED_VAL))
    print("Current weights will be saved at " + str(MODEL_WEIGHTS_PATH))
    device: torch.device = setup_device()

    ######################################## Prepare Data ########################################
    # Prepare train data
    df_train: pd.DataFrame = pd.read_csv('./data/train.csv')
    df_valid: pd.DataFrame = pd.read_csv('./data/valid.csv')

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

    # print("train_text_data: {}, train_labels: {}".format(
    #    train_text_data.count(), train_labels.count()))  # 7000, 7000

    ######################################## Tokenization & Input Formatting ########################################
    train_input_ids_and_att_masks_tuple: Tuple[List, List] = get_input_ids_and_att_masks(
        train_text_data)

    train_input_ids: torch.tensor = torch.cat(
        train_input_ids_and_att_masks_tuple[0], dim=0)
    train_attention_masks: torch.tensor = torch.cat(
        train_input_ids_and_att_masks_tuple[1], dim=0)
    train_labels: torch.tensor = torch.tensor(train_labels)

    valid_input_ids_and_att_masks_tuple: Tuple[List, List] = get_input_ids_and_att_masks(
        valid_text_data)

    valid_input_ids: torch.tensor = torch.cat(
        valid_input_ids_and_att_masks_tuple[0], dim=0)
    valid_attention_masks: torch.tensor = torch.cat(
        valid_input_ids_and_att_masks_tuple[1], dim=0)
    valid_labels: torch.tensor = torch.tensor(valid_labels)

    ######################################## Split training and feed to dataloader ########################################
    # Combine the training inputs into a TensorDataset.
    train_dataset: TensorDataset = TensorDataset(
        train_input_ids, train_attention_masks, train_labels)

    valid_dataset: TensorDataset = TensorDataset(
        valid_input_ids, valid_attention_masks, valid_labels)

    train_dataloader: DataLoader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=BATCH_SIZE  # Trains with this batch size.
    )

    validation_dataloader: DataLoader = DataLoader(
        valid_dataset,  # The validation samples.
        # Pull out batches sequentially.
        sampler=SequentialSampler(valid_dataset),
        batch_size=BATCH_SIZE  # Evaluate with this batch size.
    )

    ######################################## Initiate Model ########################################
    model = BERTweetForBinaryClassification()
    stage_1_training(model, train_dataloader,
                     validation_dataloader, device, 12, MODEL_WEIGHTS_PATH, SEED_VAL)
    # model.load_state_dict(torch.load(
    #     "BERTweet-covid19-cased-weights-1/stage_2_weights.pth", map_location=device))
    stage_2_training(model, train_dataloader,
                     validation_dataloader, device, 7, MODEL_WEIGHTS_PATH, SEED_VAL)


def final_main():
    arr = [91, 901, 9120]
    for i in range(3):
        SEED_VAL = arr[i]
        MODEL_WEIGHTS_PATH = "./BERTweet-covid19-uncased-weights-" + str(i + 9)
        main(SEED_VAL, MODEL_WEIGHTS_PATH)


if __name__ == "__main__":
    final_main()
