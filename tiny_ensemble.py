import torch
from typing import List
import numpy as np
from sklearn.metrics import classification_report


def get_classification_report(labels, preds):
    return classification_report(labels, preds)


def average_softmax(softmax_vectors: List[List[torch.tensor]]) -> List[float]:
    num_models: int = len(softmax_vectors[0])
    intermed_result: List[List[float]] = [[0, 0] for _ in range(num_models)]

    for model in softmax_vectors:
        index = 0
        for instance in model:
            intermed_result[index][0] += instance[0]
            intermed_result[index][1] += instance[1]
            index += 1

    result: List[float] = []
    for lst in intermed_result:
        result.append(np.argmax(np.asarray(lst)))

    return result


# Vote by dimension [0][0][0 or 1]
def major_voting(softmax_vectors: List[List[torch.tensor]]) -> List[float]:
    result: List[float] = []
    num_models: int = len(softmax_vectors[0])
    for model in softmax_vectors:
        instance_sum: int = 0
        for instance in model:
            instance_sum += torch.argmax(instance)
        if instance_sum == num_models / 2:
            result.append(1.0)
        else:
            result.append(float(instance_sum // (num_models / 2)))

    return result


def main() -> None:
    # Load eval softmax vectors
    bertweet_softmax: List[torch.tensor] = torch.load(
        "./softmax/BERTweet_softmax/test_softmax.pt")
    # bert_base_softmax: List[torch.tensor] = torch.load(
    #     "./softmax/BERTbase_softmax/test_softmax.pt"
    # )
    # roberta_base_softmax: List[torch.tensor] = torch.load(
    #     "./softmax/Roberta_softmax/test_softmax.pt"
    # )
    lr_softmax: List[torch.tensor] = torch.load(
        "./softmax/lr_softmax/test_softmax.pt"
    )

    softmax_vectors: List[List[torch.tensor]] = [
        bertweet_softmax,
        # bert_base_softmax,
        # roberta_base_softmax,
        lr_softmax,
    ]
    preds = average_softmax(softmax_vectors)
    labels = torch.load('./softmax/true_labels.pt')
    print(classification_report(labels, preds))


if __name__ == "__main__":
    main()
