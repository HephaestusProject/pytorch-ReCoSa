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
        self.api_config = Config.parse("./conf/api/ReCoSa.yml")
        self.recosa = RecoSAPL.load_from_checkpoint(
            self.api_config["model_path"], **self.model_config
        )
        self.data = TestDATA()

    @pytest.mark.skip(reason="To be trained.")
    def test_predict_recosa(self):
        _, res = self.recosa.predict(self.data.ctx.to(self.device))
        res_decoded = self.recosa.model.tokenizer.decode(res[0])
        self.assertEqual(res_decoded, "")


if __name__ == "__main__":
    unittest.main()
