"""
    This script was made by soeque1 at 28/07/20.
    To implement code of your network using operation from ops.py.
"""

import pytorch_lightning
from torch import nn
from torch.nn import init


class EncoderModule(nn.Module):
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
        n_layers: int,
        *args,
        **kwargs
    ) -> None:
        super(EncoderModule, self).__init__(*args, **kwargs)
        self.emb = nn.Embedding(vocab_size, embed_size)
        self.wordEnc = nn.LSTM(embed_size, hidden_size, n_layers, dropout=0.1)
        self.init_w()

    def init_w(self):
        pytorch_lightning.seed_everything(777)
        self.init_lstm()

    def init_lstm(self):
        for name, param in self.wordEnc.named_parameters():
            if "bias" in name:
                nn.init.constant(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_normal(param)
