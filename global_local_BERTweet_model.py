from transformers import RobertaConfig, RobertaModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch import nn
from typing import Tuple
import numpy as np
import torch


class BERTweetModelForClassification(BertPreTrainedModel):
    base_model_prefix = "roberta"

    def __init__(self):
        self.num_labels: int = 2
        config: RobertaConfig = RobertaConfig.from_pretrained(
            "./BERTweet_base_transformers/config.json",
            output_hidden_states=True,
        )
        super().__init__(config)
        self.bertweet: RobertaModel = RobertaModel.from_pretrained(
            "./BERTweet_base_transformers/model.bin",
            config=config
        )
        self.dense = nn.Linear(in_features=6144,
                               out_features=2048,
                               )
        self.dropout = nn.Dropout(p=0.15)
        self.dense_2 = nn.Linear(in_features=2048,
                                 out_features=512,
                                 )
        self.dense_3 = nn.Linear(in_features=512,
                                 out_features=256,
                                 )
        self.classifier = nn.Linear(in_features=256,
                                    out_features=self.num_labels,
                                    )

    def repath_input_ids_and_att_mask(self, input_ids: torch.tensor, attention_mask: torch.tensor, head_size=0.5):
        indexes: torch.tensor = (input_ids == 2).nonzero()
        max_length: int = input_ids.size()[1]
        head_repath_input_ids: List[torch.tensor] = []
        head_repath_att_mask: List[torch.tensor] = []
        tail_repath_input_ids: List[torch.tensor] = []
        tail_repath_att_mask: List[torch.tensor] = []

        for i in range(indexes.size()[0]):
            head_len = int(head_size * indexes[i][1])
            head = input_ids[i, 0:head_len].to('cpu')
            paddings: torch.tensor = torch.ones(
                (1, max_length - len(head)), dtype=torch.long)
            paddings[0, 0] = 2

            head_input_ids: torch.tensor = torch.cat(
                [torch.reshape(head, (1, -1)), paddings], dim=1)
            head_attention_masks: torch.tensor = torch.cat([torch.ones(
                (1, len(head) + 1), dtype=torch.long), torch.zeros(
                (1, max_length - len(head) - 1), dtype=torch.long)], dim=1)

            tail = input_ids[i, head_len:indexes[i][1] + 1].to('cpu')
            tail = torch.cat(
                [torch.tensor([0.], dtype=torch.long), tail], dim=0)
            paddings: torch.tensor = torch.ones(
                (1, max_length - len(tail)), dtype=torch.long)

            tail_input_ids: torch.tensor = torch.cat(
                [torch.reshape(tail, (1, -1)), paddings], dim=1)
            tail_attention_masks: torch.tensor = torch.cat([torch.ones(
                (1, len(tail)), dtype=torch.long), torch.zeros(
                (1, max_length - len(tail)), dtype=torch.long)], dim=1)

            head_repath_input_ids.append(head_input_ids)
            head_repath_att_mask.append(head_attention_masks)
            tail_repath_input_ids.append(tail_input_ids)
            tail_repath_att_mask.append(tail_attention_masks)

        device: torch.device = torch.device("cuda")
        return (
            torch.cat(head_repath_input_ids, dim=0).to(device),
            torch.cat(head_repath_att_mask, dim=0).to(device),
            torch.cat(tail_repath_input_ids, dim=0).to(device),
            torch.cat(tail_repath_att_mask, dim=0).to(device)
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        head_repath_input_ids, head_repath_att_mask, tail_repath_input_ids, tail_repath_att_mask = self.repath_input_ids_and_att_mask(
            input_ids, attention_mask, head_size=0.7)

        global_outputs = self.bertweet(
            input_ids,
            attention_mask=attention_mask,
        )
        head_outputs = self.bertweet(
            head_repath_input_ids,
            attention_mask=head_repath_att_mask
        )
        tail_outputs = self.bertweet(
            tail_repath_input_ids,
            attention_mask=tail_repath_att_mask
        )

        # Take <CLS> token for Native Layer Norm Backward
        global_hidden_states: Tuple[torch.tensor] = global_outputs[2]
        head_hidden_states: Tuple[torch.tensor] = head_outputs[2]
        tail_hidden_states: Tuple[torch.tensor] = tail_outputs[2]

        sequence_output: torch.tensor = torch.cat((
            global_hidden_states[-1][:, 0, :],
            global_hidden_states[-2][:, 0, :],
            global_hidden_states[-5][:, 0, :],
            global_hidden_states[-6][:, 0, :],
            (global_hidden_states[1][:, 0, :] +
             global_hidden_states[2][:, 0, :]) / 2.0,
            head_hidden_states[5][:, 0, :],
            head_hidden_states[0][:, 0, :],
            tail_hidden_states[0][:, 0, :]
        ), dim=1)

        sequence_output = self.dense(sequence_output)
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.dense_2(sequence_output)
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.dense_3(sequence_output)
        sequence_output = self.dropout(sequence_output)

        logits: torch.tensor = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss_function = CrossEntropyLoss()
            loss = loss_function(
                logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # loss, logits
