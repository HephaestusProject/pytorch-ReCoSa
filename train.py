"""
    This script was made by soeque1 at 24/07/20.
    To implement code for training your model.
"""

import torch
import pytorch_lightning as pl
from torch import dropout
from src.data import UbuntuDataLoader, UbuntuDataSet, collate
from src.model.net import ReCoSA, EncoderModule, DecoderModule
import torch.nn.functional as F
from pytorch_lightning import Trainer


class Config:
    vocab_size = 35000
    embed_size = 128
    utter_n_layer = 1
    dropout = .1
    output_size = 35000 # TO BE FIXED
    n_layers = 1
    hidden_size = 256
    max_position_embeddings = 50
    utter_hidden_size = 256
    decoder_hidden_size = 128

class RecoSAPL(pl.LightningModule):

    def __init__(self):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = Config()
        self.model = ReCoSA(config=config, device=device)

    def forward(self, x):
        return self.model.forward()

    def training_step(self, batch, batch_nb):
        ctx, response, cands = batch
        loss = F.cross_entropy(self.model(ctx, response), cands)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

if __name__ == "__main__":
    # train!
    data = UbuntuDataSet("./" + "tests/resources/", "sample.csv")
    dataloader = UbuntuDataLoader(data, batch_size=2, shuffle=False, num_workers=1, collate_fn=collate)

    model = RecoSAPL()
    trainer = pl.Trainer(gpus=0, num_nodes=1, distributed_backend='ddp', precision=32) #, fast_dev_run=True)
    trainer.fit(model, dataloader)
