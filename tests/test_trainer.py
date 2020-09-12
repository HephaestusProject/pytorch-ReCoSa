import unittest

import pytorch_lightning as pl
from train import RecoSAPL
from src.core.build_data import Config
from src.data import UbuntuDataLoader, UbuntuDataSet, collate


class TestReCoSa(unittest.TestCase):
    def setUp(self):
        self.data = UbuntuDataSet("./" + "tests/resources/", "sample.csv")
        self.model_config = Config.parse("./conf/model/ReCoSa.yml")
        self.trainDataLoader = UbuntuDataLoader(
            self.data, batch_size=2, shuffle=False, num_workers=1, collate_fn=collate
        )
        self.valDataLoader = UbuntuDataLoader(
            self.data, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate
        )
        self.model = RecoSAPL(self.model_config)
        self.trainer = pl.Trainer(
            gpus=0,
            num_nodes=1,
            distributed_backend="ddp_cpu",
            precision=32,
            max_epochs=1,
        )

    def test_trainer(self):
        res = self.trainer.fit(self.model, self.trainDataLoader, self.valDataLoader)


if __name__ == "__main__":
    unittest.main()
