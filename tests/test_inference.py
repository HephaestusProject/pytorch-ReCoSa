import unittest
from logging import getLogger

import pytest
import pytorch_lightning
import torch

from src.core.build_data import Config
from src.data import UbuntuDataLoader, UbuntuDataSet, collate
from train import RecoSAPL

from .test_model import SEED_NUM

logger = getLogger(__name__)


class TestReCoSaInference(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.config = Config()
        self.config.add_dataset("./conf/dataset/ubuntu_test.yml")
        self.config.add_model("./conf/model/ReCoSa_test.yml")
        self.config.add_api("./conf/api/ReCoSa.yml")
        self.recosa = RecoSAPL.load_from_checkpoint(
            self.config.api.model_path, **self.config.model
        )
        data = UbuntuDataSet(
            self.config.dataset.root + self.config.dataset.target,
            self.config.dataset.raw.train,
            self.config.model.max_seq,
            _max_turns=self.config.model.max_turns,
        )
        dataloader = UbuntuDataLoader(
            data,
            batch_size=1,
            shuffle=False,
            num_workers=8,
            collate_fn=collate,
        )
        sample = iter(dataloader)
        sample_data = next(sample)
        self.ctx = sample_data[0].to(self.device)
        self.response = sample_data[1].to(self.device)
        self.target = sample_data[2].to(self.device)
        pytorch_lightning.seed_everything(SEED_NUM)

    def test_inputs(self):
        batch_idx = 0
        self.assertEqual(
            "<|start|> thanks!  How the heck did you figure that out?. <|end|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|>",
            self.recosa.model.tokenizer.decode(self.ctx[batch_idx][-2]),
        )
        self.assertEqual(
            "<|start|> https://bugs.launchpad.net/lightdm/+bug/864109/comments/3. <|end|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|>",
            self.recosa.model.tokenizer.decode(self.ctx[batch_idx][-1]),
        )
        self.assertEqual(
            "<|start|> nice thanks!. <|end|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|>",
            self.recosa.model.tokenizer.decode(self.response[batch_idx]),
        )
        self.assertEqual(
            "nice thanks!. <|end|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|pad|> <|end|>",
            self.recosa.model.tokenizer.decode(self.target[batch_idx]),
        )

    def test_forward_recosa(self):
        dec_res = self.recosa.model(self.ctx, self.target)
        dec_res = torch.argmax(dec_res[0], dim=0)
        res_decoded = self.recosa.model.tokenizer.decode(dec_res)
        logger.debug(res_decoded)
        self.assertEqual(res_decoded.split()[0], "Thanks")

    def test_generate_recosa(self):
        _, res = self.recosa.generate(self.ctx)
        res_decoded = self.recosa.model.tokenizer.decode(res[0])
        logger.debug(res_decoded)
        self.assertEqual(res_decoded.split()[0], "Thanks,")


if __name__ == "__main__":
    unittest.main()
