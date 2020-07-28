import torch
from typing import List
import numpy as np


def average_softmax(softmax_vectors: List[List[torch.tensor]]) -> List[float]:
    num_models: int = len(softmax_vectors[0])
    intermed_result: List[List[float]] = [[] for _ in range(num_models)]

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
    print(len(bertweet_softmax))
    print(bertweet_softmax[0])
    print(len(lr_softmax))
    print(lr_softmax[0])
    print(major_voting(softmax_vectors))


if __name__ == "__main__":
    main()
