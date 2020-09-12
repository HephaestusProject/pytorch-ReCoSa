import unittest

import pytorch_lightning
import torch

from src.core.build_data import Config
from src.model.net import DecoderModule, EncoderCtxModule, EncoderResponseModule, ReCoSA

SEED_NUM = 42


class TestDATA:
    """TestDATA"""

    # batch, turn, seq_len
    ctx = torch.tensor(
        [[[3087, 4282, 2339, 2026, 4518]], [[1045, 2275, 2039, 2026, 10751]]]
    )
    # batch, seq_len
    response = torch.tensor([[3087, 4282, 2339, 2026, 4518]])
    dec_input = torch.rand([5, 1, 256])

    assert ctx.shape == torch.Size([2, 1, 5])
    assert response.shape == torch.Size([1, 5])
    assert dec_input.shape == torch.Size([5, 1, 256])


class TestCtxEncoder(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.vocab_size = 15000
        self.emb_size = 128
        self.hidden_size = 256
        self.out_size = 128
        self.n_layers = 1
        self.enc = EncoderCtxModule(
            vocab_size=self.vocab_size,
            embed_size=self.emb_size,
            hidden_size=self.hidden_size,
            out_size=self.out_size,
            n_layers=self.n_layers,
            _device=self.device,
        )
        self.data = TestDATA()
        pytorch_lightning.seed_everything(SEED_NUM)

    def test_enc(self):
        self.assertEqual(
            self.enc.rnn.weight_hh_l0.shape,
            torch.Size([4 * self.hidden_size, self.hidden_size]),
        )
        self.assertEqual(
            self.enc.rnn.weight_ih_l0.shape,
            torch.Size([4 * self.hidden_size, self.emb_size]),
        )

    def test_init(self):
        self.assertAlmostEqual(
            self.enc.rnn.weight_ih_l0[0][:5].tolist(),
            [
                -0.007601124234497547,
                0.012321769259870052,
                -0.03039667382836342,
                -0.0514763742685318,
                0.08616577088832855,
            ],
        )

        for name, param in self.enc.rnn.named_parameters():
            if "bias" in name:
                self.assertAlmostEqual(param[:5].tolist(), [0, 0, 0, 0, 0])
            elif "weight_ih_l0" in name:
                self.assertAlmostEqual(
                    param[0][:5].tolist(),
                    [
                        -0.007601124234497547,
                        0.012321769259870052,
                        -0.03039667382836342,
                        -0.0514763742685318,
                        0.08616577088832855,
                    ],
                )
            else:
                pass

    def test_input(self):
        self.assertEqual(self.data.ctx.shape, torch.Size([2, 1, 5]))

    def test_forward_enc(self):
        output = self.enc(self.data.ctx.to(self.device))
        self.assertEqual(
            output.shape,
            torch.Size(
                [self.data.ctx.shape[-1], self.data.ctx.shape[0], self.out_size]
            ),
        )


class TestResponseEncoder(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.hidden_size = 256
        self.vocab_size = 15000
        self.enc = EncoderResponseModule(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            _device=self.device,
        )
        self.data = TestDATA()
        pytorch_lightning.seed_everything(SEED_NUM)

    def test_input(self):
        self.assertEqual(self.data.response.shape, torch.Size([1, 5]))

    def test_init(self):
        self.assertAlmostEqual(
            self.enc.self_attention.in_proj_weight[0, :5].tolist(),
            [
                0.015318885445594788,
                0.048414446413517,
                -0.022115185856819153,
                0.02594170719385147,
                -0.0734342709183693,
            ],
        )

    def test_forward_enc(self):
        output = self.enc(self.data.response.to(self.device))
        self.assertEqual(
            output.shape,
            torch.Size(
                [
                    self.data.response.shape[-1],
                    self.data.response.shape[0],
                    self.hidden_size,
                ]
            ),
        )


class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.output_size = 256
        self.vocab_size = 512
        self.emb_size = 128
        self.hidden_size = 256
        self.n_layers = 1
        self.batch_size = 1
        self.dec = DecoderModule(
            output_size=self.output_size,
            embed_size=self.emb_size,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            out_size=self.vocab_size,
            _device=self.device,
        )
        self.data = TestDATA()
        pytorch_lightning.seed_everything(SEED_NUM)

    def test_forward_dec(self):
        enc_hidden = self.dec(
            self.data.dec_input.to(self.device),
            self.data.dec_input.to(self.device),
            self.data.dec_input.to(self.device),
        )
        self.assertEqual(
            enc_hidden[0].shape, torch.Size([self.batch_size, self.vocab_size])
        )


class TestReCoSa(unittest.TestCase):
    def setUp(self):
        self.config = Config.parse("./conf/model/ReCoSa.yml")
        self.device = torch.device("cpu")
        self.recosa = ReCoSA(self.config, _device=self.device)
        self.data = TestDATA()
        pytorch_lightning.seed_everything(SEED_NUM)

    def test_forward_recosa(self):
        res = self.recosa(
            self.data.ctx.to(self.device), self.data.response.to(self.device)
        )
        self.assertEqual(
            res.shape,
            torch.Size(
                [
                    self.data.ctx.size()[1],
                    self.config["vocab_size"],
                    self.data.ctx.size()[-1],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
