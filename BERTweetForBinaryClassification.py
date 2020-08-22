from transformers import RobertaConfig, RobertaModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch import nn
from typing import Tuple
import torch
import torch.nn.functional as F


class BERTweetForBinaryClassification(BertPreTrainedModel):
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
        self.dense = nn.Linear(in_features=768,
                               out_features=128,
                               )
        self.dropout = nn.Dropout(p=0.2)
        self.dense_2 = nn.Linear(in_features=128,
                                 out_features=64,
                                 )
        self.classifier = nn.Linear(in_features=64,
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
        sequence_output: torch.tensor = hidden_states[-1][:, 0, :]

        sequence_output = F.relu(self.dense(sequence_output))
        sequence_output = self.dropout(sequence_output)
        sequence_output = F.relu(self.dense_2(sequence_output))
        sequence_output = self.dropout(sequence_output)

        logits: torch.tensor = F.sigmoid(self.classifier(sequence_output))
        outputs = (logits,)
        if labels is not None:
            loss_function = CrossEntropyLoss()
            loss = loss_function(
                logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # loss, logits
