"""
    This script was made by soeque1 at 28/07/20.
    To implement code of your network using operation from ops.py.
"""

import logging
from logging import getLogger
from typing import List

import torch
from numpy.core.multiarray import concatenate
from torch import nn, tensor
from transformers import GPT2Tokenizer
from transformers.modeling_gpt2 import Block

from src.model.ops import PositionEmbedding

logger = getLogger(__name__)


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
        n_positions: int,
        padding_idx: int,
        _device: torch.device,
        *args,
        **kwargs
    ) -> None:
        super(EncoderCtxModule, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.out_size = out_size
        self.n_positions = n_positions
        self.padding_idx = padding_idx

        self.emb = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.rnn = nn.LSTM(embed_size, hidden_size, self.n_layers)  # , dropout=0.1)
        self.position_embeddings = nn.Embedding(
            n_positions, hidden_size
        )  # PositionEmbedding(hidden_size)
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

    def forward_rnn(
        self, src: List[tensor], hidden: tuple = None, _length: List = None
    ):
        embedded = self.emb(src).transpose(0, 1)
        embedded = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, _length, batch_first=False, enforce_sorted=False
        )
        if hidden is None:
            hidden = (
                torch.zeros(self.n_layers, src.shape[0], self.hidden_size).to(
                    self.device
                ),
                torch.zeros(self.n_layers, src.shape[0], self.hidden_size).to(
                    self.device
                ),
            )
        _, hidden = self.rnn(embedded, hidden)
        return hidden

    def forward(self, src: List[tensor], hidden: tuple = None):
        turn_size = src.shape[1]
        turns = []
        hidden = None
        for i in range(turn_size):
            # hidden = h_0, c_0
            turn = src[:, i, :]
            input_lengths = []
            for j_turn in turn:
                start_padding_idx = (j_turn.data == self.padding_idx).nonzero()
                if len(start_padding_idx) == 0:
                    input_lengths.append(src.shape[-1])
                else:
                    input_lengths.append(torch.min(start_padding_idx))
            input_lengths = torch.LongTensor(input_lengths)
            hidden = self.forward_rnn(turn, hidden, input_lengths)
            turns.append(hidden[0].sum(axis=0))
        turns = torch.stack(turns)
        pos = torch.cat(
            src.shape[0]
            * [torch.arange(src.shape[1], dtype=torch.long, device=self.device)]
        ).reshape(src.shape[0], -1)
        pos_emb = self.position_embeddings(pos).transpose(0, 1)
        # turns = torch.cat([turns, pos_emb], dim=-1)
        turns += pos_emb
        context, _ = self.self_attention_context(turns, turns, turns)
        turns = self.layer_norm1(turns)
        turns = self.linear1(context + turns)
        return turns


class EncoderResponseModule(nn.Module):
    """EncoderResponseModule"""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        n_positions: int,
        padding_idx: int,
        _device: torch.device,
        *args,
        **kwargs
    ) -> None:
        super(EncoderResponseModule, self).__init__()
        self.device = _device
        self.hidden_size = hidden_size
        self.n_positions = n_positions
        self.emb = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.position_embeddings = nn.Embedding(
            n_positions, hidden_size
        )  # PositionEmbedding(hidden_size)
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
        embedded = self.emb(src)
        pos = torch.cat(
            src.shape[0]
            * [torch.arange(self.n_positions, dtype=torch.long, device=self.device)]
        ).reshape(src.shape[0], -1)
        pos_emb = self.position_embeddings(pos)
        # response = torch.cat([embedded, pos_emb], dim=-1)
        # response = torch.cat(
        #     [embedded, pos_emb[:, : embedded.shape[1], :]], dim=-1
        # ).transpose(0, 1)
        response = (embedded + pos_emb[:, : embedded.shape[1], :]).transpose(0, 1)
        mask = self._generate_square_subsequent_mask(src.shape[1]).to(self.device)
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
        self.embed = nn.Embedding(output_size, embed_size)
        self.self_attention = nn.MultiheadAttention(hidden_size, 8)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, out_size)

        self.init_w()
        self.device = _device

    def init_w(self):
        pass

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, key, query, value, _train=True):
        if _train:
            seq_len = query.shape[0]
            mask = self._generate_square_subsequent_mask(seq_len)[:, : key.shape[0]].to(
                self.device
            )
            output, _ = self.self_attention(query, key, value, attn_mask=mask)
        else:
            # query seq >= key seq
            if query.shape[0] >= key.shape[0]:
                seq_len = query.shape[0]
                mask = self._generate_square_subsequent_mask(seq_len)[
                    :, : key.shape[0]
                ].to(self.device)
            else:
                seq_len = key.shape[0]
                mask = self._generate_square_subsequent_mask(seq_len)[
                    : query.shape[0]
                ].to(self.device)
            output, _ = self.self_attention(query, key, value, attn_mask=mask)
        output = self.layer_norm1(output)
        output = self.linear1(output)
        return output


class ReCoSA(nn.Module):
    def __init__(self, config: dict, _device: torch.device) -> None:
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained("./conf/tokenizer")
        self.vocab_size = config["vocab_size"]
        assert len(self.tokenizer) == self.vocab_size

        self._device = _device
        self.encoderCtx = EncoderCtxModule(
            vocab_size=config["vocab_size"],
            embed_size=config["embed_size"],
            hidden_size=config["utter_hidden_size"],
            out_size=config["out_size"],
            n_layers=config["utter_n_layer"],
            n_positions=config["max_turns"],
            padding_idx=self.tokenizer.pad_token_id,
            _device=_device,
        )

        self.encoderResponse = EncoderResponseModule(
            vocab_size=config["vocab_size"],
            hidden_size=config["out_size"],
            n_positions=config["max_seq"],
            padding_idx=self.tokenizer.pad_token_id,
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

    def forward(self, ctx: torch.Tensor, response: torch.Tensor, _train: bool = True):
        enc_ctx = self.encoderCtx(ctx)
        dec_res = self.inference(enc_ctx, response, _train)
        return dec_res

    def inference(
        self, enc_ctx: torch.Tensor, response: torch.Tensor, _train: bool
    ) -> torch.Tensor:
        enc_res = self.encoderResponse(response)
        dec_res = self.decoder(query=enc_res, key=enc_ctx, value=enc_ctx, _train=_train)
        # batch, vocab_size, seq_len
        dec_res = dec_res.permute(1, 2, 0)
        logger.debug(self.tokenizer.decode(torch.argmax(dec_res[0], dim=0)))
        return dec_res

    def predict(self, ctx: torch.Tensor, batch_size: int = 1, max_seq: int = 50) -> str:
        bos_token_id = self.tokenizer.bos_token_id
        # eos_token_id = self.tokenizer.eos_token_id
        pred_res_max = torch.tensor([[bos_token_id]] * batch_size).to(self._device)
        enc_ctx = self.encoderCtx(ctx)
        for _ in range(max_seq):
            pred_res = self.inference(enc_ctx, pred_res_max, _train=False)
            current_pred_res_max = torch.argmax(pred_res[:, :, -1], dim=1).unsqueeze(1)
            pred_res_max = torch.cat([pred_res_max, current_pred_res_max], dim=1)
        return pred_res, pred_res_max[:, 1:]

    def decode(self, response: List[torch.Tensor]) -> str:
        return self.tokenizer.decode(response)

    def tie_weights(self):
        """"""
        self._tie_or_clone_weights(self.decoder.linear1, self.decoder.embed)

    def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
        """Tie or clone module weights depending of whether we are using TorchScript or not"""
        if self.config.torchscript:
            output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
        else:
            output_embeddings.weight = input_embeddings.weight

        if getattr(output_embeddings, "bias", None) is not None:
            output_embeddings.bias.data = torch.nn.functional.pad(
                output_embeddings.bias.data,
                (
                    0,
                    output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                ),
                "constant",
                0,
            )
        if hasattr(output_embeddings, "out_features") and hasattr(
            input_embeddings, "num_embeddings"
        ):
            output_embeddings.out_features = input_embeddings.num_embeddings


if __name__ == "__main__":
    pass
