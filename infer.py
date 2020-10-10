"""
    This script was made by soeque1 at 24/07/20.
    To implement code for inference with your model.
"""

from logging import getLogger
from typing import List

import pytorch_lightning
import torch

from src.data import make_max_contexts
from src.model.net import ReCoSA
from tests.test_model import SEED_NUM
from train import RecoSAPL

logger = getLogger(__name__)


class ReCoSaAPI:
    def __init__(self, config: dict) -> None:
        self.recosa = RecoSAPL.load_from_checkpoint(
            config.api.model_path, **config.model
        )
        self.recosa.model.eval()
        pytorch_lightning.seed_everything(SEED_NUM)
        self._max_history = config.model.max_turns
        self._max_seq = config.model.max_seq

    def encode_fn(self, _input: str) -> List[int]:
        """encode

        Args:
            _input (str): str

        Returns:
            List[int]: tokenize.encode({bos} {input} {eos})
        """
        _input = "{bos} {sentence} {eos}".format(
            bos=self.recosa.model.tokenizer.bos_token,
            sentence=_input,
            eos=self.recosa.model.tokenizer.eos_token,
        )
        return self.recosa.model.tokenizer.encode(
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
        _, res = self.recosa.model.generate(_ctx)
        res_decoded = self.recosa.model.tokenizer.decode(res[0])
        return res_decoded.split(self.recosa.model.tokenizer.eos_token)[0]


if __name__ == "__main__":
    pass
