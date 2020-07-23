from transformers import RobertaConfig, RobertaModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch import nn
from torchvision import models
import torch


device: torch.device = torch.device("cuda")


class BERTweetForBinaryClassification(BertPreTrainedModel):
    base_model_prefix = "roberta"

    def __init__(self):
        self.num_labels: int = 2
        config: RobertaConfig = RobertaConfig.from_pretrained(
            "./BERTweet_base_transformers/config.json"
        )
        super().__init__(config)
        self.model: RobertaModel = RobertaModel.from_pretrained(
            "./BERTweet_base_transformers/model.bin",
            config=config
        )
        self.resnet = models.resnet18(pretrained=True)
        num_final_in = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features=num_final_in,
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
        sequence_output: torch.tensor = outputs[0][:, 0, :]

        batch_size = sequence_output.size()[0]
        vector_length = sequence_output.size()[1]
        sequence_output = torch.reshape(
            sequence_output, (batch_size, 1, vector_length, 1))

        zeros: torch.tensor = torch.zeros(
            (batch_size, 1, vector_length, 1)).to(device)

        sequence_output = torch.cat(
            [torch.cat([sequence_output, zeros], dim=1), zeros], dim=1)

        logits: torch.tensor = self.resnet(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss_function = CrossEntropyLoss()
            loss = loss_function(
                logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # loss, logits
