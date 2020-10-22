"""
    This script was made by soeque1 at 24/07/20.
    To implement code for inference with your model.
"""

from logging import getLogger
from typing import List

import pytorch_lightning
import torch

from src.data import make_max_contexts
from tests.test_model import SEED_NUM

logger = getLogger(__name__)


class Predictor:
    def __init__(self, model, tokenizer, _max_history: int, _max_seq: int) -> None:
        pytorch_lightning.seed_everything(SEED_NUM)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self._max_history = _max_history
        self._max_seq = _max_seq

    @classmethod
    def from_checkpoint(cls, PLModel, config: dict):
        recosa = PLModel.load_from_checkpoint(
            checkpoint_path=config.api.model_path, config=config
        )
        return cls(
            model=recosa.model,
            tokenizer=recosa.model.tokenizer,
            _max_history=config.model.max_turns,
            _max_seq=config.model._max_seq,
        )

    def encode_fn(self, _input: str) -> List[int]:
        """encode

        Args:
            _input (str): str

        Returns:
            List[int]: tokenize.encode({bos} {input} {eos})
        """
        _input = "{bos} {sentence} {eos}".format(
            bos=self.tokenizer.bos_token,
            sentence=_input,
            eos=self.tokenizer.eos_token,
        )
        return self.tokenizer.encode(
            _input,
            add_special_tokens=False,
            max_length=self._max_seq,
            padding="max_length",
            truncation=True,
        )

    def encode_ctx(self, ctx: list) -> torch.Tensor:
        """encoding contenxts

        Args:
            ctx (list): [description]

        Returns:
            torch.Tensor: [description]
        """
        return [self.encode_fn(i) for i in ctx]

    def generate(self, _input: str) -> str:
        # seq_len, turns
        _ctx = torch.tensor(
            self.encode_ctx(make_max_contexts(_input.split("\n"), self._max_history))
        )
        # seq_len, batch, turns
        _ctx = torch.unsqueeze(_ctx, 0)
        _, res = self.model.generate(_ctx)
        res_decoded = self.tokenizer.decode(res[0])
        return res_decoded.split(self.tokenizer.eos_token)[0]


if __name__ == "__main__":
    pass
