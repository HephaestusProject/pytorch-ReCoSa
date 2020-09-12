import unittest

import torch

from src.core.build_data import Config
from src.data import UbuntuDataLoader, UbuntuDataSet, collate


class TestDataSet(unittest.TestCase):
    def setUp(self):
        self.data_config = Config.parse("./conf/dataset/ubuntu_test.yml")
        self.model_config = Config.parse("./conf/model/ReCoSa.yml")
        self.data = UbuntuDataSet(
            folderpath=self.data_config["root"] + self.data_config["target"],
            filepath=self.data_config["raw"]["train"],
        )
        self.test_batch_size = 2

    def test_config(self):
        self.assertEqual(list(self.data_config.keys()), ["root", "target", "raw"])

    def test_model(self):
        self.assertEqual(
            list(self.model_config.keys()),
            [
                "output_size",
                "vocab_size",
                "embed_size",
                "utter_hidden_size",
                "utter_n_layer",
                "batch_size",
                "dropout",
                "out_size",
                "max_seq",
                "max_turns",
            ],
        )

    def test_dataset(self):
        instance_len = 3  # ctxs, response, target
        instance = self.data[0]
        ctxs = instance[0]
        response = instance[1]
        target = instance[2]

        # instance
        self.assertEqual(instance_len, len(instance))

        # ctx
        self.assertEqual(self.model_config["max_turns"], len(ctxs))

        for ctx in ctxs:
            self.assertEqual(self.model_config["max_seq"], len(ctx))

        # response
        self.assertEqual(self.model_config["max_seq"], len(response))

        # target
        self.assertEqual(self.model_config["max_seq"], len(target))

    def test_dataloader(self):
        dataloader = UbuntuDataLoader(
            self.data,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collate,
        )
        for (ctx, response, cands) in dataloader:
            self.assertEqual(
                ctx.shape,
                torch.Size(
                    [
                        self.test_batch_size,
                        self.model_config["max_turns"],
                        self.model_config["max_seq"],
                    ]
                ),
            )
            self.assertEqual(
                response.shape,
                torch.Size([self.test_batch_size, self.model_config["max_seq"]]),
            )
            self.assertEqual(
                cands.shape,
                torch.Size([self.test_batch_size, self.model_config["max_seq"]]),
            )
            break


if __name__ == "__main__":
    unittest.main()
