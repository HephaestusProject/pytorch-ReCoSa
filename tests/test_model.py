import unittest
import torch
from src.model.net import EncoderModule


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 50
        self.emb_size = 10
        self.hidden_size = 30
        self.n_layers = 2
        self.enc = EncoderModule(
            vocab_size=self.vocab_size,
            embed_size=self.emb_size,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
        )

    def test_enc(self):
        self.assertEqual(self.enc.wordEnc.weight_hh_l0.shape, torch.Size([4 * self.hidden_size, self.hidden_size]))
        self.assertEqual(self.enc.wordEnc.weight_ih_l0.shape, torch.Size([4 * self.hidden_size, self.emb_size]))

    def test_init(self):
        self.assertAlmostEqual(self.enc.wordEnc.weight_ih_l0[0][:5].tolist(), [0.023242833092808723, -0.012155871838331223, -0.04731405898928642, -0.10140256583690643, 0.19815635681152344])

        for name, param in self.enc.wordEnc.named_parameters():
            if "bias" in name:
                self.assertAlmostEqual(param[:5].tolist(), [0, 0, 0, 0, 0])
            elif "weight_ih_l0" in name:
                self.assertAlmostEqual(param[0][:5].tolist(), [0.023242833092808723, -0.012155871838331223, -0.04731405898928642, -0.10140256583690643, 0.19815635681152344])
            else:
                pass


if __name__ == "__main__":
    unittest.main()
