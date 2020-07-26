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
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Tuple, List, Callable

from transformers import BertModel, BertTokenizer


def dataload(_path:str):
    data = []
    with open(_path, 'r', newline='') as read:
        csv_read = csv.reader(read)
        next(csv_read)  # eat header

        for line in csv_read:
            fields = [
                s.replace('__eou__', '.').replace('__eot__', '\n').strip()
                for s in line
            ]
            context = fields[0]
            response = fields[1]
            cands = None
            if len(fields) > 3:
                cands = [fields[i] for i in range(2, len(fields))]
                cands.append(response)
                random.shuffle(cands)
            data.append([context, [response], cands])
    return data


class UbuntuDataSet(Dataset):
    """
        Ubuntu DataSet
    """
    def __init__(self, folderpath:str, filepath: str) -> None:
        """

        """
        self._path = folderpath + '/' + filepath
        self._corpus = dataload(self._path)

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._corpus[idx]


class UbuntuDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(UbuntuDataLoader, self).__init__(*args, **kwargs)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', unk_token='<|unkwn|>')

from torch.nn.utils.rnn import pad_sequence
def collate(examples):
    input_ids = [tokenizer.encode(i, add_special_tokens=False, max_length=10, pad_to_max_length=True) for i in examples[0]]
    token_type_ids = [tokenizer.encode(i, add_special_tokens=False, max_length=10, pad_to_max_length=True) for i in examples[1]]
    attention_masks = [tokenizer.encode(i, add_special_tokens=False, max_length=10, pad_to_max_length=True) for i in examples[2]]
    print(attention_masks)
    return torch.stack([input_ids, token_type_ids, attention_masks])


if __name__ == '__main__':
    pass