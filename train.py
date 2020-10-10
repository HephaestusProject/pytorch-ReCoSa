"""
    This script was made by soeque1 at 24/07/20.
    To implement code for training your model.
"""

import logging
from argparse import ArgumentParser, Namespace
from logging import getLogger

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.core.build_data import Config
from src.data import UbuntuDataLoader, UbuntuDataSet, collate
from src.metric import bleuS
from src.model.net import ReCoSA
from src.utils.prepare import build

logger = getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RecoSAPL(pl.LightningModule):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ReCoSA(config=self.config, _device=self._device)
        self.pred = []
        self.target = []

    def forward(self, x):
        return self.model.forward()

    def inference(self, ctx: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
        return self.model.inference(ctx, response)

    def generate(self, ctx: torch.Tensor) -> str:
        return self.model.generate(ctx)

    def training_step(self, batch, batch_idx):
        ctx, response, target = batch
        pred = self.model(ctx, response)
        if batch_idx % 1000 == 0:
            logger.info(self.model.tokenizer.decode(torch.argmax(pred[0], dim=0)))
            logger.info(self.model.tokenizer.decode(response[0]))
        loss = F.cross_entropy(
            pred, target, ignore_index=self.model.tokenizer.pad_token_id
        )
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
        loss = F.cross_entropy(
            pred, target, ignore_index=self.model.tokenizer.pad_token_id
        )
        ppl = torch.exp(loss)
        result = pl.EvalResult(checkpoint_on=loss)

        if batch_idx % 100 == 0:
            pred_sen = torch.argmax(pred, dim=1)
            pred_sentence = [
                self.model.tokenizer.decode(i)
                .split(self.model.tokenizer.eos_token)[0]
                .split()
                + [self.model.tokenizer.eos_token]
                for i in pred_sen
            ]
            target_sentence = [
                self.model.tokenizer.decode(i)
                .split(self.model.tokenizer.eos_token)[0]
                .split()
                + [self.model.tokenizer.eos_token]
                for i in target
            ]
            logger.info("idx: " + str(batch_idx))
            logger.info("pred: " + " ".join(pred_sentence[0]))
            logger.info("target: " + " ".join(target_sentence[0]))

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
        optimizer = Adam(params=self.model.parameters(), lr=1e-5)
        lr_scheduler = ReduceLROnPlateau(
            optimizer, patience=10, factor=0.9, verbose=True
        )
        # reduce every epoch (default)
        scheduler = {
            "scheduler": lr_scheduler,
            "reduce_on_plateau": True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            "monitor": "val_checkpoint_on",
        }
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        ctx, _, target = batch
        pred, pred_sen = self.model.generate(ctx, max_seq=ctx.shape[2])
        loss = F.cross_entropy(
            pred, target, ignore_index=self.model.tokenizer.pad_token_id
        )
        ppl = torch.exp(loss)
        result = pl.EvalResult(checkpoint_on=loss)
        pred_sentence = [
            self.model.tokenizer.decode(i)
            .split(self.model.tokenizer.eos_token)[0]
            .split()
            + [self.model.tokenizer.eos_token]
            for i in pred_sen
        ]
        target_sentence = [
            self.model.tokenizer.decode(i)
            .split(self.model.tokenizer.eos_token)[0]
            .split()
            + [self.model.tokenizer.eos_token]
            for i in target
        ]
        target_sentence_list = [[i] for i in target_sentence]
        self.pred.extend(pred_sentence)
        self.target.extend(target_sentence_list)
        bleu_score = bleuS(pred_sentence, target_sentence_list).to(ppl.device)

        if batch_idx % 10 == 0:
            logger.info("idx: " + str(batch_idx))
            ctx_decoded = [
                self.model.tokenizer.decode(i).split(self.model.tokenizer.eos_token)[0]
                + self.model.tokenizer.eos_token
                for i in ctx[0]
            ]
            logger.info("idx: " + " ".join(ctx_decoded))
            logger.info("pred: " + " ".join(pred_sentence[0]))
            logger.info("target: " + " ".join(target_sentence[0]))

        result.log_dict(
            {"val_loss_gen": loss, "val_ppl_gen": ppl, "val_bleu_gen": bleu_score},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return result


def main(
    config_data_file: str,
    config_model_file: str,
    config_trainer_file: str,
    version: str,
) -> None:

    # TODO: to be removed
    _ = build({"data_config": config_data_file, "version": version})
    cfg = Config()
    cfg.add_dataset(config_data_file)
    cfg.add_model(config_model_file)
    cfg.add_trainer(config_trainer_file)

    train_data = UbuntuDataSet(
        cfg.dataset.root + cfg.dataset.target,
        cfg.dataset.raw.train,
        cfg.model.max_seq,
        cfg.dataset.target,
        cfg.model.max_turns,
    )
    val_data = UbuntuDataSet(
        cfg.dataset.root + cfg.dataset.target,
        cfg.dataset.raw.val,
        cfg.model.max_seq,
        cfg.dataset.target,
        cfg.model.max_turns,
    )
    train_dataloader = UbuntuDataLoader(
        train_data,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate,
    )
    val_dataloader = UbuntuDataLoader(
        val_data,
        batch_size=cfg.model.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate,
    )
    logger = TensorBoardLogger(save_dir="exp", name=cfg.dataset.target, version=version)

    prefix = f"exp/{cfg.dataset.target}/{version}/"
    suffix = "{epoch:02d}-{val_loss:.4f}"
    filepath = prefix + suffix
    checkpoint_callback = ModelCheckpoint(
        filepath=filepath,
        save_top_k=1,
        monitor="val_loss",
        save_weights_only=True,
        verbose=True,
    )

    model = RecoSAPL(cfg.model)
    lr_logger = LearningRateLogger(logging_interval="step")
    trainer = pl.Trainer(
        **cfg.trainer.pl,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks=[lr_logger],
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
    parser.add_argument(
        "--config_trainer_file", default="./conf/trainer/ReCoSa.yml", type=str
    )
    parser.add_argument("--version", default="v0.0.1", type=str)
    args = parser.parse_args()
    main(
        args.config_data_file,
        args.config_model_file,
        args.config_trainer_file,
        args.version,
    )
