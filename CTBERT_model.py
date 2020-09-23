from transformers import BertConfig, BertModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch import nn
from typing import Tuple
import torch

MODEL_PATH: str = "./CTBERT/"


class CTBERTForBinaryClassification(BertPreTrainedModel):
    base_model_prefix = "bert"

    def __init__(self):
        self.num_labels: int = 2
        config: BertConfig = BertConfig.from_pretrained(
            MODEL_PATH + "config.json",
            output_hidden_states=True,
        )
        super().__init__(config)
        self.model: BertModel = BertModel.from_pretrained(
            MODEL_PATH + "pytorch_model.bin",
            config=config
        )
        self.dense = nn.Linear(in_features=1024,
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

        sequence_output = self.dense(sequence_output)
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.dense_2(sequence_output)
        sequence_output = self.dropout(sequence_output)

        logits: torch.tensor = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss_function = CrossEntropyLoss()
            loss = loss_function(
                logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # loss, logits
