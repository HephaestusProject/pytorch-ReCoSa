import pytest
import torch
import unittest
from train import RecoSAPL
from src.core.build_data import Config
from src.data import UbuntuDataLoader, UbuntuDataSet, collate


class TestReCoSaInference(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        config_data = Config.parse("./conf/dataset/ubuntu_test.yml")
        config_model = Config.parse("./conf/model/ReCoSa.yml")
        config_api = Config.parse("./conf/api/ReCoSa.yml")
        self.recosa = RecoSAPL.load_from_checkpoint(
            config_api["model_path"], **config_model
        )
        data = UbuntuDataSet(
            config_data["root"] + config_data["target"],
            config_data["raw"]["train"],
            config_model["max_seq"],
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

    def test_inputs(self):
        batch_idx = 0
        self.assertEqual(
            "[CLS] anyone knows why my stock oneiric exports env var'username '? i mean what is that used for? i know of $ user but not $ username. my precise install doesn't export username. [SEP] [PAD] [PAD]",
            self.recosa.model.tokenizer.decode(self.ctx[batch_idx][0]),
        )
        self.assertEqual(
            '[CLS] looks like it used to be exported by lightdm, but the line had the comment " / / fixme : is this required? " so i guess it isn\'t surprising it is gone. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]',
            self.recosa.model.tokenizer.decode(self.ctx[batch_idx][1]),
        )
        self.assertEqual(
            "[CLS] nice thanks!. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]",
            self.recosa.model.tokenizer.decode(self.response[batch_idx]),
        )
        self.assertEqual(
            "nice thanks!. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]",
            self.recosa.model.tokenizer.decode(self.target[batch_idx]),
        )

    # @pytest.mark.skip(reason="To be trained.")
    def test_predict_recosa(self):
        _, res = self.recosa.predict(self.ctx)
        res_decoded = self.recosa.model.tokenizer.decode(res[0])
        print(res_decoded)
        self.assertEqual(res_decoded.split()[0], "thanks")


if __name__ == "__main__":
    unittest.main()
