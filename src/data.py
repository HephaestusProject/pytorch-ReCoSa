# Copyright (c) HephaestusProject
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
    This script was made by soeque1 at 24/07/20.
    To implement code for data pipeline. (e.g. custom class subclassing torch.utils.data.Dataset)
"""

import csv
import random
from typing import Callable, List, Tuple

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
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
            context = fields[0].strip() #[i.strip() for i in fields[0].split('\n')]
            response = fields[1].strip()
            cands = None
            if len(fields) > 3:
                cands = [fields[i] for i in range(2, len(fields))]
                cands.append(response)
                random.shuffle(cands)
            data.append({"context": context, "response": response, "cands": cands})
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
        self._tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", unk_token="<|unkwn|>"
        )
        self.max_length = 50

    def __len__(self) -> int:
        return len(self._corpus)

    def encode_fn(self, _input: str) -> List[int]:
        return self._tokenizer.encode(
            _input,
            add_special_tokens=False,
            max_length=self.max_length,
            pad_to_max_length=True,
        )

    def get_cands_for_retreival(self, cands: list):
        return [self.encode_fn(i) for i in cands]

    def get_cands_for_generation(self, cands: list):
        return self.encode_fn(cands[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset = self._corpus[idx]
        ctx = self.encode_fn(dataset["context"])
        response = self.encode_fn(dataset["response"])
        cands = self.get_cands_for_generation(dataset["cands"])
        return (ctx, response, cands)


class UbuntuDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(UbuntuDataLoader, self).__init__(*args, **kwargs)


class Tokenizer(BertTokenizer):
    def __init__(self, vocab_file: str, *args, **kwargs):
        super(Tokenizer, self).__init__(vocab_file, *args, **kwargs)

    # vocab_files_names = {"vocab_file": "vocab.txt"}
    # pretrained_vocab_files_map = {"vocab_file": {"bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"}}
    # max_model_input_sizes = {"bert-base-uncased": 512}
    # pretrained_init_configuration = {"bert-base-uncased": {"do_lower_case": True}}
    # model_input_names = ["attention_mask"]


def collate(examples):
    return list(map(torch.LongTensor, zip(*examples)))


if __name__ == "__main__":
    pass
