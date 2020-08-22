"""
    This script was made by soeque1 at 28/07/20.
    To implement code of your network using operation from ops.py.
"""

import math
import random
from typing import List, Optional

import pytorch_lightning
import torch
from torch import dropout, nn, tensor
from torch.functional import Tensor
from torch.nn import init

SEED_NUM = 777
initrange = 0.1


class PositionEmbedding(nn.Module):

    """
    Position embedding for self-attention
    refer: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    d_model: word embedding size or output size of the self-attention blocks
    max_len: the max length of the input squeezec
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # [1, max_len]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)  # not the parameters of the Module

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class EncoderCtxModule(nn.Module):
    """
    ReCoSa model.
    See https://arxiv.org/pdf/1907.05339 for more details
    1. wordEnc: LSTM
    2. ctxSA: Context Self-Attention
    3. resSA: Response Self-Attention
    4. ctxResSA: Context-Response Self-Attention
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int,
        hidden_size: int,
        out_size: int,
        n_layers: int,
        _device: torch.device,
        *args,
        **kwargs
    ) -> None:
        super(EncoderCtxModule, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.out_size = out_size

        self.emb = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.LSTM(embed_size, hidden_size, self.n_layers, dropout=0.1)
        self.position_embeddings = PositionEmbedding(hidden_size)
        self.self_attention_context = nn.MultiheadAttention(hidden_size, 8)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, out_size)

        self.init_w()
        self.device = _device

    def init_w(self):
        pytorch_lightning.seed_everything(SEED_NUM)
        self.init_lstm()

    def init_lstm(self):
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal(param)

    def forward_rnn(self, src: List[tensor], hidden: tuple = None):
        embedded = self.emb(src)
        if not hidden:
            hidden = (
                torch.randn(self.n_layers, src.shape[-1], self.hidden_size).to(
                    self.device
                ),
                torch.randn(self.n_layers, src.shape[-1], self.hidden_size).to(
                    self.device
                ),
            )
        _, hidden = self.rnn(embedded, hidden)
        return hidden

    def forward(self, src: List[tensor]):
        turn_size = len(src)
        turns = []
        for i in range(turn_size):
            h_0, c_0 = self.forward_rnn(src[i])
            turns.append(h_0.sum(axis=0))
        turns = torch.stack(turns)
        turns = turns.transpose(0, 1)

        turns = self.position_embeddings(turns)
        context, _ = self.self_attention_context(turns, turns, turns)
        turns = self.layer_norm1(turns)
        turns = self.linear1(context + turns)
        return turns


class EncoderResponseModule(nn.Module):
    """EncoderResponseModule
    """

    def __init__(
        self, vocab_size: int, hidden_size: int, _device: torch.device, *args, **kwargs
    ) -> None:
        super(EncoderResponseModule, self).__init__()
        self.device = _device
        self.hidden_size = hidden_size
        self.emb = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = PositionEmbedding(hidden_size)
        self.self_attention = nn.MultiheadAttention(hidden_size, 8)
        self.init_w()

    def init_w(self):
        pytorch_lightning.seed_everything(SEED_NUM)
        self.self_attention.in_proj_weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src: List[tensor]):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
        Shape:
            see the docs in Transformer class.
        """
        src = src.transpose(0, 1)
        embedded = self.emb(src)
        response = self.position_embeddings(embedded)
        mask = self._generate_square_subsequent_mask(len(src)).to(self.device)
        attn, _ = self.self_attention(response, response, response, attn_mask=mask)
        return attn


class DecoderModule(nn.Module):
    """
    ReCoSa model.
    See https://arxiv.org/pdf/1907.05339 for more details
    1. wordEnc: LSTM
    2. ctxSA: Context Self-Attention
    3. resSA: Response Self-Attention
    4. ctxResSA: Context-Response Self-Attention
    """

    def __init__(
        self,
        output_size: int,
        embed_size: int,
        hidden_size: int,
        out_size: int,
        _device: torch.device,
        *args,
        **kwargs
    ) -> None:
        super(DecoderModule, self).__init__()
        self.embed = nn.Embedding(output_size, embed_size, padding_idx=0)
        self.self_attention = nn.MultiheadAttention(hidden_size, 8)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, out_size)

        self.init_w()
        self.device = _device

    def init_w(self):
        pytorch_lightning.seed_everything(SEED_NUM)

    def forward(self, key, query, value):
        output, _ = self.self_attention(query, key, value)
        output = self.layer_norm1(output)
        output = self.linear1(output)
        return output


class ReCoSA(nn.Module):
    def __init__(self, config: dict, _device: torch.device) -> None:
        import pdb; pdb.set_trace()
        super().__init__()
        self.encoderCtx = EncoderCtxModule(
            vocab_size=config["vocab_size"],
            embed_size=config["embed_size"],
            hidden_size=config["utter_hidden_size"],
            out_size=config["out_size"],
            n_layers=config["utter_n_layer"],
            _device=_device,
        )

        self.encoderResponse = EncoderResponseModule(
            vocab_size=config["vocab_size"],
            hidden_size=config["out_size"],
            _device=_device,
        )

        self.decoder = DecoderModule(
            output_size=config["output_size"],
            embed_size=config["embed_size"],
            hidden_size=config["out_size"],
            out_size=config["vocab_size"],
            n_layers=config["utter_n_layer"],
            _device=_device,
        )

        self.init()

    def init(self):
        pytorch_lightning.seed_everything(SEED_NUM)
        self.encoderCtx.init_w()
        self.encoderResponse.init_w()
        self.decoder.init_w()

    def forward(self, ctx: torch.Tensor, response: torch.Tensor):
        enc_ctx = self.encoderCtx(ctx)
        enc_res = self.encoderResponse(response)
        # seq_len, batch, vocab_size
        dec_res = self.decoder(query=enc_res, key=enc_ctx, value=enc_ctx)
        # batch, vocab_size, seq_len
        dec_res = dec_res.permute(1, 2, 0)
        return dec_res

    def predict(self, ctx: torch.Tensor, response: torch.Tensor):
        assert NotImplementedError


if __name__ == "__main__":
    pass
