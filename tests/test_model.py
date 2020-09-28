import unittest

import pytest
import pytorch_lightning
import torch

from src.core.build_data import Config
from src.model.net import (DecoderModule, EncoderCtxModule,
                           EncoderResponseModule, ReCoSA)

SEED_NUM = 42


class TestDATA:
    """TestDATA"""

    # batch, turn, seq_len
    ctx = torch.tensor(
        [
            [
                [101, 3087, 4282, 2339, 2026, 4518, 102],
                [101, 1045, 2275, 2039, 2026, 10751, 102],
            ]
        ]
    )
    # batch, seq_len
    response = torch.tensor([[101, 3087, 4282, 2339, 2026, 4518, 102]])
    dec_input = torch.rand([7, 1, 256])

    # batch, turn, seq_len
    assert ctx.shape == torch.Size([1, 2, 7])
    # batch, seq_len
    assert response.shape == torch.Size([1, 7])
    # seq_len, batch, hidden_size
    assert dec_input.shape == torch.Size([7, 1, 256])


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
                -0.04221362620592117,
                -0.052751462906599045,
                0.0416913703083992,
                0.014486987143754959,
                0.029713977128267288,
            ],
        )

        for name, param in self.enc.rnn.named_parameters():
            if "bias" in name:
                self.assertAlmostEqual(param[:5].tolist(), [0, 0, 0, 0, 0])
            elif "weight_ih_l0" in name:
                self.assertAlmostEqual(
                    param[0][:5].tolist(),
                    [
                        -0.04221362620592117,
                        -0.052751462906599045,
                        0.0416913703083992,
                        0.014486987143754959,
                        0.029713977128267288,
                    ],
                )
            else:
                pass

    def test_input(self):
        self.assertEqual(self.data.ctx.shape, torch.Size([1, 2, 7]))

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
        self.assertEqual(self.data.response.shape, torch.Size([1, 7]))

    def test_init(self):
        self.assertListEqual(
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
        self.config = Config()
        self.config.add_model("./conf/model/ReCoSa_test.yml")
        self.device = torch.device("cpu")
        self.recosa = ReCoSA(self.config.model, _device=self.device)
        self.data = TestDATA()
        pytorch_lightning.seed_everything(SEED_NUM)

    def test_input(self):
        # ctxs
        # turns, batch, seq_len
        # 1th-turn
        self.assertEqual(
            "[CLS] anyone knows why my stock [SEP]",
            self.recosa.tokenizer.decode(self.data.ctx[0, 0, :]),
        )
        # 2nd-turn
        self.assertEqual(
            "[CLS] i set up my hd [SEP]",
            self.recosa.tokenizer.decode(self.data.ctx[0, 1, :]),
        )

    def test_forward_recosa(self):
        res = self.recosa(
            self.data.ctx.to(self.device), self.data.response.to(self.device)
        )
        self.assertEqual(
            res.shape,
            torch.Size(
                [
                    self.data.ctx.size()[0],
                    self.config.model.vocab_size,
                    self.data.ctx.size()[-1],
                ]
            ),
        )

    def test_predict_recosa(self):
        ctxs = self.data.ctx.to(self.device)
        res, res_max = self.recosa.predict(ctxs, max_seq=ctxs.shape[2])
        batch_size = self.data.ctx.shape[0]
        self.assertEqual(
            torch.Size([batch_size, self.config.model.vocab_size, ctxs.shape[2]]),
            res.shape,
        )
        self.assertEqual(torch.Size([batch_size, ctxs.shape[2]]), res_max.shape)


if __name__ == "__main__":
    unittest.main()
