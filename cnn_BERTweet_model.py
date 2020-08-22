from transformers import RobertaConfig, RobertaModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch import nn
from typing import Tuple
import torch
import torch.nn.functional as F


class BERTweetModelForClassification(BertPreTrainedModel):
    base_model_prefix = "roberta"

    def __init__(self):
        self.num_labels: int = 2
        config: RobertaConfig = RobertaConfig.from_pretrained(
            "./BERTweet_base_transformers/config.json",
            output_hidden_states=True,
        )
        super().__init__(config)
        self.model: RobertaModel = RobertaModel.from_pretrained(
            "./BERTweet_base_transformers/model.bin",
            config=config
        )

        # CAN YOU ADD A CNN LAYER HERE?
        self.dense = nn.Linear(in_features=767,
                               out_features=128,
                               )
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(in_features=128,
                                    out_features=self.num_labels,
                                    )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
        )
        # Take <CLS> token for Native Layer Norm Backward
        hidden_states: Tuple[torch.tensor] = outputs[2]
        last_sequence_output: torch.tensor = hidden_states[-1][:, 0, :]
        second_to_last_sequence_output: torch.tensor = hidden_states[-2][:, 0, :]
        third_to_last_sequence_output: torch.tensor = hidden_states[-3][:, 0, :]
        fourth_to_last_sequence_output: torch.tensor = hidden_states[-4][:, 0, :]
        fifth_to_last_sequence_output: torch.tensor = hidden_states[-5][:, 0, :]
        sixth_to_last_sequence_output: torch.tensor = hidden_states[-6][:, 0, :]
        seventh_to_last_sequence_output: torch.tensor = hidden_states[-7][:, 0, :]
        eighth_to_last_sequence_output: torch.tensor = hidden_states[-8][:, 0, :]
        nineth_to_last_sequence_output: torch.tensor = hidden_states[-9][:, 0, :]
        tenth_to_last_sequence_output: torch.tensor = hidden_states[-10][:, 0, :]
        eleventh_to_last_sequence_output: torch.tensor = hidden_states[-11][:, 0, :]
        first_sequence_output: torch.tensor = hidden_states[0][:, 0, :]

        sequence_output: torch.tensor = torch.stack((
            last_sequence_output,
            second_to_last_sequence_output,
            third_to_last_sequence_output,
            fourth_to_last_sequence_output,
            fifth_to_last_sequence_output,
            sixth_to_last_sequence_output,
            seventh_to_last_sequence_output,
            eighth_to_last_sequence_output,
            nineth_to_last_sequence_output,
            tenth_to_last_sequence_output,
            eleventh_to_last_sequence_output,
            first_sequence_output
        ), dim=1)

        sequence_output = F.relu(self.dense(sequence_output))
        sequence_output = self.dropout(sequence_output)

        logits: torch.tensor = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss_function = CrossEntropyLoss()
            loss = loss_function(
                logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # loss, logits
