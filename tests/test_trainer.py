import unittest

import pytorch_lightning as pl

from src.core.build_data import Config
from src.data import UbuntuDataLoader, UbuntuDataSet, collate
from train import RecoSAPL


class TestReCoSa(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.config.add_model("./conf/model/ReCoSa_test.yml")
        self.config.add_trainer("./conf/trainer/ReCoSa_test.yml")
        self.data = UbuntuDataSet(
            "./" + "tests/resources/Ubuntu/",
            "sample.csv",
            self.config.model.max_seq,
            "Ubuntu",
            self.config.model.max_turns,
        )
        self.trainDataLoader = UbuntuDataLoader(
            self.data,
            **self.config.trainer.data,
            collate_fn=collate,
        )
        self.valDataLoader = UbuntuDataLoader(
            self.data,
            **self.config.trainer.data,
            collate_fn=collate,
        )
        self.model = RecoSAPL(self.config, len(self.data))
        self.trainer = pl.Trainer(**self.config.trainer.pl)

    def test_trainer(self):
        self.assertFalse(self.config.trainer.data.shuffle)
        _ = self.trainer.fit(self.model, self.trainDataLoader, self.valDataLoader)


if __name__ == "__main__":
    unittest.main()
