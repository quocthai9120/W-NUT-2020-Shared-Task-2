import torch
import argparse

from typing import List

from transformers import RobertaConfig
from transformers import RobertaModel

from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary


def get_bert_embedding(lines: List[str]) -> List[torch.Tensor]:
    # Load model
    config = RobertaConfig.from_pretrained(
        "./BERTweet_base_transformers/config.json"
    )
    BERTweet = RobertaModel.from_pretrained(
        "./BERTweet_base_transformers/model.bin",
        config=config
    )

    # Load BPE encoder
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpe-codes',
                        default="./BERTweet_base_transformers/bpe.codes",
                        required=False,
                        type=str,
                        help='path to fastBPE BPE'
                        )
    args = parser.parse_args()
    bpe = fastBPE(args)

    # Load the dictionary
    vocab = Dictionary()
    vocab.add_from_file("./BERTweet_base_transformers/dict.txt")

    result: List[torch.Tensor] = []
    for i in range(len(lines)):
        line: str = lines[i]

        # Encode the line using fastBPE & Add prefix <s> and suffix </s>
        subwords = '<s> ' + bpe.encode(line) + ' </s>'

        # Map subword tokens to corresponding indices in the dictionary
        input_ids = vocab.encode_line(
            subwords, append_eos=False, add_if_not_exist=False).long().tolist()

        # Convert into torch tensor
        all_input_ids = torch.tensor([input_ids], dtype=torch.long)

        features = None

        with torch.no_grad():
            features = BERTweet(all_input_ids)

        result.append(features[0][:, 0, :].numpy()[0])

    return result
