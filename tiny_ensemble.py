import torch
from typing import List


def average_softmax(softmax_vectors: List[List[torch.tensor]]) -> List[float]:
    pass


def major_voting(softmax_vectors: List[List[torch.tensor]]) -> List[float]:
    pass


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
        bert_base_softmax,
        roberta_base_softmax,
        lr_softmax,
    ]

    print(average_softmax(softmax_vectors))
    print(major_voting(softmax_vectors))


if __name__ == "__main__":
    main()
