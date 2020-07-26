# Copyright (c) HephaestusProject
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
    This script was made by soeque1 at 24/07/20.
    To implement code for data pipeline. (e.g. custom class subclassing torch.utils.data.Dataset)
"""

import csv
import random
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Tuple, List, Callable

from transformers import BertModel, BertTokenizer


def dataload(_path: str) -> List[str]:
    data = []
    with open(_path, "r", newline="") as read:
        csv_read = csv.reader(read)
        next(csv_read)  # eat header

        for line in csv_read:
            fields = [
                s.replace("__eou__", ".").replace("__eot__", "\n").strip() for s in line
            ]
            context = fields[0]
            response = fields[1]
            cands = None
            if len(fields) > 3:
                cands = [fields[i] for i in range(2, len(fields))]
                cands.append(response)
                random.shuffle(cands)
            data.append({'context': context, 'response': response, 'cands': cands})
    return data


class UbuntuDataSet(Dataset):
    """
        Ubuntu DataSet
        TODO: Cache
    """

    def __init__(self, folderpath: str, filepath: str) -> None:
        """

        """
        self._path = folderpath + "/" + filepath
        self._corpus = dataload(self._path)

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._corpus[idx]


class UbuntuDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(UbuntuDataLoader, self).__init__(*args, **kwargs)


class Tokenizer(BertTokenizer):
    def __init__(self, vocab_file:str, *args, **kwargs):
        super(Tokenizer, self).__init__(vocab_file, *args, **kwargs)
    # vocab_files_names = {"vocab_file": "vocab.txt"}
    # pretrained_vocab_files_map = {"vocab_file": {"bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"}}
    # max_model_input_sizes = {"bert-base-uncased": 512}
    # pretrained_init_configuration = {"bert-base-uncased": {"do_lower_case": True}}
    # model_input_names = ["attention_mask"]

# TODO: Refactor
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", unk_token="<|unkwn|>")

# TODO: Move to Dataset
def collate(examples):
    # examples: B X 3
    ctx = [
        tokenizer.encode(
            example['context'], add_special_tokens=False, max_length=20, pad_to_max_length=True
        )
        for example in examples
    ]
    response = [
        tokenizer.encode(
            example['response'], add_special_tokens=False, max_length=20, pad_to_max_length=True
        )
        for example in examples
    ]

    cands = []
    for example in examples:
        tokenized = []
        for ex_cands in example['cands']:
            tokenized.append(tokenizer.encode(
            ex_cands, add_special_tokens=False, max_length=20, pad_to_max_length=True
        ))
        cands.append(tokenized)
    # B X S
    ctx = torch.tensor(ctx)
    # B X S
    response = torch.tensor(response)
    # B X C * S
    cands = torch.tensor(cands)

    return ctx, response, cands


if __name__ == "__main__":
    pass
