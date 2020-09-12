"""
    This script was made by soeque1 at 24/07/20.
    To implement code for training your model.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torch.nn.utils.rnn import pad_sequence

from src.core.build_data import Config
from src.data import UbuntuDataLoader, UbuntuDataSet, collate
from src.model.net import ReCoSA
from src.utils.prepare import build

from argparse import ArgumentParser, Namespace


class RecoSAPL(pl.LightningModule):
    def __init__(
        self, sep_token_id: int, config_path: str = "./conf/model/ReCoSa.yml"
    ) -> None:
        super().__init__()
        self.config = Config.parse(config_path)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ReCoSA(config=self.config, _device=self._device)

    def forward(self, x):
        return self.model.forward()

    def training_step(self, batch, batch_idx):
        ctx, response, cands = batch
        loss = F.cross_entropy(self.model(ctx, response), cands, ignore_index=0)
        result = pl.TrainResult(minimize=loss)
        result.log_dict(
            {"tr_loss": loss},
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return result

    def validation_step(self, batch, batch_idx):
        ctx, response, cands = batch
        loss = F.cross_entropy(self.model(ctx, response), cands, ignore_index=0)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict(
            {"val_loss": loss},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def main(config_data_file: str, version: str):
    config = build({"config": config_data_file, "version": version})
    train_data = UbuntuDataSet(
        config["root"] + config["target"], config["raw"]["train"]
    )
    val_data = UbuntuDataSet(config["root"] + config["target"], config["raw"]["val"])

    train_dataloader = UbuntuDataLoader(
        train_data, batch_size=256, shuffle=False, num_workers=8, collate_fn=collate
    )
    val_dataloader = UbuntuDataLoader(
        val_data, batch_size=256, shuffle=False, num_workers=8, collate_fn=collate
    )
    logger = TensorBoardLogger(save_dir="exp", name=config["target"], version=version)

    prefix = f"exp/{config['target']}/{version}/"
    suffix = "{epoch:02d}-{avg_tr_loss:.4f}-{avg_tr_acc:.4f}-{avg_val_loss:.4f}-{avg_val_acc:.4f}"
    filepath = prefix + suffix
    checkpoint_callback = ModelCheckpoint(
        filepath=filepath,
        save_top_k=1,
        monitor="avg_val_acc",
        save_weights_only=True,
        verbose=True,
    )

    model = RecoSAPL(train_data._tokenizer.sep_token_id)

    trainer = pl.Trainer(
        gpus=1,
        num_nodes=1,
        distributed_backend="ddp",
        amp_level="O2",
        amp_backend="native",
        max_epochs=10,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_data_file", default="./conf/dataset/ubuntu.yml", type=str
    )
    parser.add_argument("--version", default="v0.0.1", type=str)
    args = parser.parse_args()
    main(args.config_data_file, args.version)