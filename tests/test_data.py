import unittest

import torch

from src.core.build_data import Config
from src.data import UbuntuDataLoader, UbuntuDataSet, collate


class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.config = Config.parse("./conf/dataset/ubuntu.yml")
        self.data = UbuntuDataSet("./" + "tests/resources/", "sample.csv")
        self.batch_size = 2
        self.seq_len = 50

    def test_dataset(self):
        dataloader = UbuntuDataLoader(
            self.data, batch_size=2, shuffle=False, num_workers=1, collate_fn=collate
        )
        for (ctx, response, cands) in dataloader:
            self.assertEqual(ctx.shape, torch.Size([self.batch_size, self.seq_len]))
            self.assertEqual(
                response.shape, torch.Size([self.batch_size, self.seq_len])
            )
            self.assertEqual(cands.shape, torch.Size([self.batch_size, self.seq_len]))
            break


if __name__ == "__main__":
    unittest.main()
