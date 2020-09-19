"""
    This script was made by soeque1 at 24/07/20.
    To implement code for training your model.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.metrics.nlp import BLEUScore

from src.core.build_data import Config
from src.data import UbuntuDataLoader, UbuntuDataSet, collate
from src.model.net import ReCoSA
from src.utils.prepare import build

from argparse import ArgumentParser, Namespace


class RecoSAPL(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ReCoSA(config=self.config, _device=self._device)

    def forward(self, x):
        return self.model.forward()

    def inference(self, ctx: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        return self.model.inference(ctx, response)

    def predict(self, ctx: torch.Tensor) -> str:
        return self.model.predict(ctx)

    def training_step(self, batch, batch_idx):
        ctx, response, target = batch
        pred = self.model(ctx, response)
        if batch_idx % 1000 == 0:
            print(self.model.tokenizer.decode(torch.argmax(pred[0], dim=0)))
            print(self.model.tokenizer.decode(response[0]))
        loss = F.cross_entropy(pred, target, ignore_index=0)
        ppl = torch.exp(loss)
        result = pl.TrainResult(minimize=loss)
        result.log_dict(
            {"tr_loss": loss, "tr_ppl": ppl},
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return result

    def validation_step(self, batch, batch_idx):
        ctx, response, target = batch
        pred = self.model(ctx, response)
        loss = F.cross_entropy(pred, target, ignore_index=0)
        ppl = torch.exp(loss)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict(
            {"val_loss": loss, "val_ppl": ppl},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return result

    def configure_optimizers(self):
        opt = Adam(params=self.model.parameters(), lr=1e-4)
        scheduler = ExponentialLR(opt, gamma=0.95)
        return [opt], [scheduler]

    def test_step(self, batch, batch_idx):
        bleuS = BLEUScore()
        ctx, _, target = batch
        pred, pred_max = self.model.predict(
            ctx, batch_size=ctx.shape[0], max_seq=ctx.shape[2]
        )
        loss = F.cross_entropy(pred, target, ignore_index=0)
        ppl = torch.exp(loss)
        result = pl.EvalResult(checkpoint_on=loss)
        pred_sentence = [self.model.tokenizer.decode(i).split() for i in pred_max]
        target_senctence = [self.model.tokenizer.decode(i).split() for i in target]
        bleu_score = bleuS(pred_sentence, target_senctence).to(ppl.device)

        # TO BE FIXED:
        # PPL, BLEU (for corpus)
        result.log_dict(
            {"val_loss_gen": loss, "val_ppl_gen": ppl, "val_bleu_gen": bleu_score},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return result


def main(config_data_file: str, config_model_file: str, version: str) -> None:
    config_data = build({"config": config_data_file, "version": version})
    config_model = Config.parse(config_model_file)

    train_data = UbuntuDataSet(
        config_data["root"] + config_data["target"],
        config_data["raw"]["train"],
        config_model["max_seq"],
        config_data["target"],
    )
    val_data = UbuntuDataSet(
        config_data["root"] + config_data["target"],
        config_data["raw"]["val"],
        config_model["max_seq"],
        config_data["target"],
    )

    train_dataloader = UbuntuDataLoader(
        train_data,
        batch_size=config_model["batch_size"],
        shuffle=True,
        num_workers=8,
        collate_fn=collate,
    )
    val_dataloader = UbuntuDataLoader(
        val_data,
        batch_size=config_model["batch_size"],
        shuffle=False,
        num_workers=8,
        collate_fn=collate,
    )
    logger = TensorBoardLogger(
        save_dir="exp", name=config_data["target"], version=version
    )

    prefix = f"exp/{config_data['target']}/{version}/"
    suffix = "{epoch:02d}-{val_loss:.4f}"
    filepath = prefix + suffix
    checkpoint_callback = ModelCheckpoint(
        filepath=filepath,
        save_top_k=1,
        monitor="val_loss",
        save_weights_only=True,
        verbose=True,
    )

    model = RecoSAPL(config_model)

    trainer_params = {
        "gpus": [0],
        "num_nodes": 1,
        "distributed_backend": "ddp",
        "amp_level": "O2",
        "amp_backend": "native",
    }

    trainer = pl.Trainer(
        **trainer_params,
        max_epochs=100,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_data_file", default="./conf/dataset/ubuntu.yml", type=str
    )
    parser.add_argument(
        "--config_model_file", default="./conf/model/ReCoSa.yml", type=str
    )
    parser.add_argument("--version", default="v0.0.1", type=str)
    args = parser.parse_args()
    main(args.config_data_file, args.config_model_file, args.version)