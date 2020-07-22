from transformers import RobertaConfig, RobertaModel, BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from torch import nn
from torchvision import models


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
<<<<<<< HEAD

        # self.classifier = model

        self.resnet = models.resnet18(pretrained=True)
        num_final_in = self.resnet.fc.in_features
        self.classifier = nn.Linear(in_features=num_final_in,
=======
        self.dense = nn.Linear(in_features=768,
                               out_features=64,
                               )
        self.dropout = nn.Dropout(p=0.2)
        self.dense_2 = nn.Linear(in_features=64,
                                 out_features=64,
                                 )
        self.classifier = nn.Linear(in_features=64,
>>>>>>> de55464ffe68e9d5ed43f624eefef2ed53dc2721
                                    out_features=self.num_labels,
                                    )

# model = models.resnet18(pretrained = True)
num_final_in = model.fc.in_features
NUM_CLASSES = 1
model.fc = nn.Linear(num_final_in, NUM_CLASSES)

# Code for freezing different layers
ct = 0
for child in model.children():
    ct += 1
    if ct < 4:
        for param in child.parameters():
            param.require_grad = False
            
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
<<<<<<< HEAD
        sequence_output = self.resnet(sequence_output)
=======
        sequence_output = self.dense(sequence_output)
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.dense_2(sequence_output)
        sequence_output = self.dropout(sequence_output)
>>>>>>> de55464ffe68e9d5ed43f624eefef2ed53dc2721
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss_function = CrossEntropyLoss()
            loss = loss_function(
                logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # loss, logits
