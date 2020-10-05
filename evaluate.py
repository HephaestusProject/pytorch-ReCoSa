"""
    This script was made by soeque1 at 24/07/20.
    To implement code for training your model.
"""

import logging
from argparse import ArgumentParser, Namespace
from logging import getLogger

import pytorch_lightning as pl
from pytorch_lightning.metrics.nlp import BLEUScore

from src.core.build_data import Config
from src.data import UbuntuDataLoader, UbuntuDataSet, collate
from src.utils.prepare import build
from train import RecoSAPL

logger = getLogger(__name__)


def main(
    config_data_file: str,
    config_model_file: str,
    config_trainer_file: str,
    config_api_file: str,
    version: str,
) -> None:

    # TODO: to be removed
    _ = build({"data_config": config_data_file, "version": version})

    cfg = Config()
    cfg.add_dataset(config_data_file)
    cfg.add_model(config_model_file)
    cfg.add_api(config_api_file)
    cfg.add_trainer(config_trainer_file)

    val_data = UbuntuDataSet(
        cfg.dataset.root + cfg.dataset.target,
        cfg.dataset.raw.val,
        cfg.model.max_seq,
        cfg.dataset.target,
        cfg.model.max_turns,
    )

    val_dataloader = UbuntuDataLoader(
        val_data,
        batch_size=cfg.model.batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate,
    )

    model = RecoSAPL.load_from_checkpoint(cfg.api.model_path, **cfg.model)
    cfg.trainer.pl.max_epochs = 1

    trainer = pl.Trainer(**cfg.trainer.pl, logger=False, checkpoint_callback=False)
    test_result = trainer.test(model, test_dataloaders=val_dataloader)
    logger.info(test_result)
    bleuS = BLEUScore(n_gram=4, smooth=True)
    bleu_score = bleuS(model.pred, model.target)
    logger.info(bleu_score)


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
    parser.add_argument("--config_api_file", default="./conf/api/ReCoSa.yml", type=str)
    parser.add_argument("--version", default="v0.0.8.1", type=str)
    args = parser.parse_args()
    main(
        args.config_data_file,
        args.config_model_file,
        args.config_trainer_file,
        args.config_api_file,
        args.version,
    )
