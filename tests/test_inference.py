import pytest
import torch
import unittest
from train import RecoSAPL
from src.core.build_data import Config
from .test_model import TestDATA


class TestReCoSaInference(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model_config = Config.parse("./conf/model/ReCoSa.yml")
        self.recosa = RecoSAPL.load_from_checkpoint(
            "exp/Ubuntu/v0.0.2/epoch=05-val_loss=6.0326.ckpt", **self.model_config
        )
        self.data = TestDATA()

    def test_inference_recosa(self):
        bos_token_id = 101
        res = self.recosa.inference(
            self.data.ctx.to(self.device),
            torch.tensor([[bos_token_id]]).to(self.device),
        )
        res_decoded = self.recosa.model.tokenizer.decode([res.item()])
        self.assertEqual(".", res_decoded)

    @pytest.mark.skip(reason="To be trained.")
    def test_predict_recosa(self):
        res = self.recosa.predict(self.data.ctx.to(self.device))
        self.assertEqual(res, "")


if __name__ == "__main__":
    unittest.main()
