"""
    This script was made by soeque1 at 28/07/20.
    To implement code of your network using operation from ops.py.
"""

import random
from typing import List, Optional

import pytorch_lightning
import torch
from torch import dropout, nn, tensor
from torch.functional import Tensor
from torch.nn import init
from transformers import BertTokenizer

from src.model.ops import PositionEmbedding


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
        self.init_lstm()

    def init_lstm(self):
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal(param)

    def forward_rnn(self, src: List[tensor], hidden: tuple = None):
        embedded = self.emb(src)
        if hidden is None:
            hidden = (
                torch.zeros(self.n_layers, src.shape[-1], self.hidden_size).to(
                    self.device
                ),
                torch.zeros(self.n_layers, src.shape[-1], self.hidden_size).to(
                    self.device
                ),
            )
        _, hidden = self.rnn(embedded, hidden)
        return hidden

    def forward(self, src: List[tensor], hidden: tuple = None):
        turn_size = len(src)
        turns = []
        hidden = None
        for i in range(turn_size):
            # hidden = h_0, c_0
            hidden = self.forward_rnn(src[i], hidden)
            turns.append(hidden[0].sum(axis=0))
        turns = torch.stack(turns)
        turns = turns.transpose(0, 1)

        turns = self.position_embeddings(turns)
        context, _ = self.self_attention_context(turns, turns, turns)
        turns = self.layer_norm1(turns)
        turns = self.linear1(context + turns)
        return turns


class EncoderResponseModule(nn.Module):
    """EncoderResponseModule"""

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
        initrange = 0.1
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
        pass

    def forward(self, key, query, value):
        output, _ = self.self_attention(query, key, value)
        output = self.layer_norm1(output)
        output = self.linear1(output)
        return output


class ReCoSA(nn.Module):
    def __init__(self, config: dict, _device: torch.device) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.vocab_size = config["vocab_size"]
        assert len(self.tokenizer) == self.vocab_size

        self._device = _device
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
        self.encoderCtx.init_w()
        self.encoderResponse.init_w()
        self.decoder.init_w()

    def forward(self, ctx: torch.Tensor, response: torch.Tensor):
        enc_ctx = self.encoderCtx(ctx)
        dec_res = self.inference(enc_ctx, response)
        return dec_res

    def inference(self, enc_ctx: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        enc_res = self.encoderResponse(response)
        dec_res = self.decoder(query=enc_res, key=enc_ctx, value=enc_ctx)
        # batch, vocab_size, seq_len
        dec_res = dec_res.permute(1, 2, 0)
        return dec_res

    def predict(self, ctx: torch.Tensor, batch_size: int = 1, max_seq: int = 50) -> str:
        bos_token_id = 101
        eos_token_id = 102
        pred_res_max = torch.tensor([[bos_token_id]] * batch_size).to(self._device)
        enc_ctx = self.encoderCtx(ctx)
        for _ in range(max_seq):
            pred_res = self.inference(enc_ctx, pred_res_max)
            current_pred_res_max = torch.argmax(pred_res[:, :, 0], dim=1).unsqueeze(1)
            pred_res_max = torch.cat([pred_res_max, current_pred_res_max], dim=1)
            # TODO: batch_size
            # if pred_res_max[0].tolist() == eos_token_id:
            #     break
        return pred_res, pred_res_max[:, 1:]

    def decode(self, response: List[torch.Tensor]) -> str:
        return self.tokenizer.decode(response)


if __name__ == "__main__":
    pass
