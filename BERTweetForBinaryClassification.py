from transformers import RobertaConfig, RobertaModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch import nn


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
        self.dropout = nn.Dropout()
        self.dense = nn.Linear(in_features=768,
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
        sequence_output = outputs[0][:, 0, :]
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.dense(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss_function = CrossEntropyLoss()
            loss = loss_function(
                logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # loss, logits
