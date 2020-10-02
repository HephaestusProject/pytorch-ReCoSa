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
from transformers import GPT2Tokenizer

from src.core.build_data import Config


def make_max_contexts(ctx: list, _max_history: int):
    context = []
    for idx in range(_max_history):
        try:
            context.append(ctx[idx])
        except IndexError:
            context.append("")
    return context


def dataload_ubuntu(_path: str, _max_history: int = 5) -> List[str]:
    data = []
    with open(_path, "r", newline="") as read:
        csv_read = csv.reader(read)
        next(csv_read)  # eat header

        for line in csv_read:
            fields = [
                s.replace("__eou__", ".").replace("__eot__", "\n").strip() for s in line
            ]

            context = make_max_contexts(fields[0].split("\n"), _max_history)
            try:
                response = fields[1].strip()
            except IndexError:
                import pdb

                pdb.set_trace()
            cands = None
            if len(fields) > 3:
                cands = [fields[i] for i in range(2, len(fields))]
                cands.append(response)
                random.shuffle(cands)
            data.append({"context": context, "response": response, "cands": cands})

    return data


def dataload_DSTC7_AVSD(_path: str, _max_history: int = 5) -> List[str]:
    """https://github.com/gmftbyGMFTBY/MultiTurnDialogZoo/blob/master/data/dataset_process/plato_process.py

    Args:
        _path (str): [description]
        _max_history (int, optional): [description]. Defaults to 5.

    Returns:
        List[str]: [description]
    """
    data = []
    with open(_path, "r", newline="") as read:
        csv_read = csv.reader(read, delimiter="\n")

        for line in csv_read:
            context = []
            response = []
            dialogue = line[0].strip()
            se = dialogue.split("\t")
            assert len(se) == 3

            # PersonaChat
            knowledge, ctx, res = se
            ku = knowledge.split("__eou__")
            ku = ["<user0> " + i.strip() for i in ku]
            ku = " __eou__ ".join(ku)
            cu = ctx.split("__eou__")

            speaker = "<user0> "
            fcu = []
            for i in cu:
                fcu.append(speaker + i.strip())
                speaker = "<user0> " if speaker == "<user1> " else "<user1> "
            fcu = " __eou__ ".join(fcu)
            # train/dev is different from the test in DSTC7_AVSD dataset
            if "|" in res:
                res = speaker + res.split("|")[0].strip()
            else:
                res = speaker + res.strip()
            context = make_max_contexts(
                (ku + " __eou__ " + fcu).split("__eou__"), _max_history
            )
            response = res
            cands = None
            data.append({"context": context, "response": response, "cands": cands})

    return data


class UbuntuDataSet(Dataset):
    """
    Ubuntu DataSet
    TODO: Cache
    """

    def __init__(
        self,
        folderpath: str,
        filepath: str,
        _max_seq: int = 50,
        _data_name: str = "Ubuntu",
    ) -> None:
        """"""
        self._path = folderpath + "/" + filepath
        if _data_name == "Ubuntu":
            self._corpus = dataload_ubuntu(self._path)
        elif _data_name == "DSTC7_AVSD":
            self._corpus = dataload_DSTC7_AVSD(self._path)
        else:
            raise NotImplementedError
        self._tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self._tokenizer.bos_token = "<|start|>"
        self._tokenizer.eos_token = "<|end|>"
        self._tokenizer.pad_token = "<|pad|>"
        self._tokenizer.unk_token = "<|unk|>"
        self._tokenizer.add_tokens(
            [
                self._tokenizer.bos_token,
                self._tokenizer.eos_token,
                self._tokenizer.pad_token,
                self._tokenizer.unk_token,
            ]
        )
        self._tokenizer.padding_side = "right"
        self._tokenizer.save_pretrained("./conf/tokenizer")
        self._max_seq = _max_seq

    def __len__(self) -> int:
        return len(self._corpus)

    def encode_fn(self, _input: str) -> List[int]:
        _input = "{bos} {sentence} {eos}".format(
            bos=self._tokenizer.bos_token,
            sentence=_input,
            eos=self._tokenizer.eos_token,
        )
        return self._tokenizer.encode(
            _input,
            add_special_tokens=False,
            max_length=self._max_seq,
            padding="max_length",
            truncation=True,
        )

    def get_ctx(self, ctx: list) -> torch.Tensor:
        return [self.encode_fn(i) for i in ctx]

    def get_cands_for_retreival(self, cands: list) -> List[torch.Tensor]:
        return [self.encode_fn(i) for i in cands]

    def get_cands_for_generation(self, response: str) -> torch.Tensor:
        return self.encode_fn(response)[1:] + [self._tokenizer.eos_token_id]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset = self._corpus[idx]
        ctx = self.get_ctx(dataset["context"])
        response = self.encode_fn(dataset["response"])
        target = self.get_cands_for_generation(dataset["response"])
        return (ctx, response, target)


class UbuntuDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(UbuntuDataLoader, self).__init__(*args, **kwargs)


class Tokenizer(GPT2Tokenizer):
    def __init__(self, vocab_file: str, *args, **kwargs):
        super(Tokenizer, self).__init__(vocab_file, *args, **kwargs)


def collate(examples):
    return list(map(torch.LongTensor, zip(*examples)))


if __name__ == "__main__":
    pass
