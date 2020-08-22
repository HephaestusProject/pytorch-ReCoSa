"""
    This script was made by soeque1 at 24/07/20.
    To implement code for training your model.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from torch.nn.utils.rnn import pad_sequence

from src.core.build_data import Config
from src.data import UbuntuDataLoader, UbuntuDataSet, collate
from src.model.net import ReCoSA


class RecoSAPL(pl.LightningModule):
    def __init__(self, sep_token_id: int, config_path: str="./conf/model/ReCoSa.yml") -> None:
        super().__init__()
        self.config = Config.parse(config_path)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ReCoSA(config=self.config, _device=self._device)
        self.max_turns = self.config["max_turns"]
        self.sep_token_id = torch.Tensor([sep_token_id]).to(self._device)

    def forward(self, x):
        return self.model.forward()

    def training_step(self, batch, batch_idx):
        ctx, response, cands = batch
        turns = (
            torch.zeros([response.shape[0], self.max_turns, response.shape[1]])
            .type(torch.LongTensor)
            .to(self._device)
        )

        for batch_idx, i in enumerate(ctx):
            split_idx = (i == self.sep_token_id).nonzero()[:, 0]
            if split_idx.shape >= torch.Size([1]):
                pair_split_idx = torch.cat(
                    [
                        torch.cat([torch.tensor([0]).to(self._device), split_idx]),
                        torch.tensor([i.shape[0]]).to(self._device),
                    ]
                )
            else:
                pair_split_idx = torch.tensor([0, i.shape[0]]).to(self._device)
            for idx in range(len(pair_split_idx) - 1):
                turns[
                    batch_idx, idx, pair_split_idx[idx] : pair_split_idx[idx + 1]
                ] = i[pair_split_idx[idx] : pair_split_idx[idx + 1]]

        loss = F.cross_entropy(self.model(turns, response), cands, ignore_index=0)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, on_epoch=True)
        return result

    def validation_step(self, batch, batch_idx):
        ctx, response, cands = batch
        turns = (
            torch.zeros([response.shape[0], self.max_turns, response.shape[1]])
            .type(torch.LongTensor)
            .to(self._device)
        )

        for batch_idx, i in enumerate(ctx):
            split_idx = (i == self.sep_token_id).nonzero()[:, 0]
            if split_idx.shape >= torch.Size([1]):
                pair_split_idx = torch.cat(
                    [
                        torch.cat([torch.tensor([0]).to(self._device), split_idx]),
                        torch.tensor([i.shape[0]]).to(self._device),
                    ]
                )
            else:
                pair_split_idx = torch.tensor([0, i.shape[0]]).to(self._device)
            for idx in range(len(pair_split_idx) - 1):
                turns[
                    batch_idx, idx, pair_split_idx[idx] : pair_split_idx[idx + 1]
                ] = i[pair_split_idx[idx] : pair_split_idx[idx + 1]]

        loss = F.cross_entropy(self.model(turns, response), cands, ignore_index=0)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == "__main__":
    # train!
    data = UbuntuDataSet("./" + "tests/resources/", "sample.csv")
    dataloader = UbuntuDataLoader(
        data, batch_size=2, shuffle=False, num_workers=1, collate_fn=collate
    )
    model = RecoSAPL(data._tokenizer.sep_token_id)
    trainer = pl.Trainer(
        gpus=0, num_nodes=1, distributed_backend="ddp", precision=32, max_epochs=1
    )  # , fast_dev_run=True)
    trainer.fit(model, dataloader)
