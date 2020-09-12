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

from src.core.build_data import Config


def dataload(_path: str, _max_history: int = 5) -> List[str]:
    def make_max_contexts(ctx: list, _max_history: int):
        context = []
        for idx in range(_max_history):
            try:
                context.append(ctx[idx])
            except IndexError:
                context.append("")
        return context

    data = []
    with open(_path, "r", newline="") as read:
        csv_read = csv.reader(read)
        next(csv_read)  # eat header

        for line in csv_read:
            fields = [
                s.replace("__eou__", ".").replace("__eot__", "\n").strip() for s in line
            ]

            context = make_max_contexts(fields[0].split("\n"), _max_history)
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
        """"""
        self._path = folderpath + "/" + filepath
        self._corpus = dataload(self._path)
        self._config = Config.parse("./conf/model/ReCoSa.yml")
        self._tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", unk_token="<|unkwn|>"
        )
        self.max_length = self._config["max_seq"]
        self.vocab_size = len(self._tokenizer)

    def __len__(self) -> int:
        return len(self._corpus)

    def encode_fn(self, _input: str) -> List[int]:
        return self._tokenizer.encode(
            _input,
            add_special_tokens=True,
            max_length=self.max_length,
            pad_to_max_length=True,
        )

    def get_ctx(self, ctx: list) -> torch.Tensor:
        return [self.encode_fn(i) for i in ctx]

    def get_cands_for_retreival(self, cands: list) -> List[torch.Tensor]:
        return [self.encode_fn(i) for i in cands]

    def get_cands_for_generation(self, cands: list) -> torch.Tensor:
        return self.encode_fn(cands[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset = self._corpus[idx]
        ctx = self.get_ctx(dataset["context"])
        response = self.encode_fn(dataset["response"])
        cands = self.get_cands_for_generation(dataset["cands"])
        return (ctx, response, cands)


class UbuntuDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(UbuntuDataLoader, self).__init__(*args, **kwargs)


class Tokenizer(BertTokenizer):
    def __init__(self, vocab_file: str, *args, **kwargs):
        super(Tokenizer, self).__init__(vocab_file, *args, **kwargs)


def collate(examples):
    return list(map(torch.LongTensor, zip(*examples)))


if __name__ == "__main__":
    pass
