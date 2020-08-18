"""
    This script was made by soeque1 at 24/07/20.
    To implement code for training your model.
"""

import torch
import pytorch_lightning as pl
from src.core.build_data import Config
from src.data import UbuntuDataLoader, UbuntuDataSet, collate
from src.model.net import ReCoSA
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pytorch_lightning import Trainer


class RecoSAPL(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.config = Config.parse("./conf/model/ReCoSa.yml")
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ReCoSA(config=self.config, _device=self._device)
        self.max_turns = self.config['max_turns']

    def forward(self, x):
        return self.model.forward()

    def training_step(self, batch, batch_nb):
        ctx, response, cands = batch
        turns = torch.zeros([response.shape[0], self.max_turns, response.shape[1]]).type(torch.LongTensor).to(self._device)

        for batch_idx, i in enumerate(ctx):
            split_idx = (i == data._tokenizer.sep_token_id).nonzero()[:, 0]
            if split_idx.shape >= torch.Size([1]):
                pair_split_idx = torch.cat([torch.cat([torch.tensor([0]), split_idx]), torch.tensor([i.shape[0]])])
            else:
                pair_split_idx = torch.tensor([0, i.shape[0]])
            for idx in range(len(pair_split_idx) - 1):
                turns[batch_idx, idx, pair_split_idx[idx]: pair_split_idx[idx +1]] = i[pair_split_idx[idx]: pair_split_idx[idx +1]]

        loss = F.cross_entropy(self.model(turns, response), cands, ignore_index=0)
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
