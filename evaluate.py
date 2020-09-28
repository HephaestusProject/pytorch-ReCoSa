"""
    This script was made by soeque1 at 24/07/20.
    To implement code for training your model.
"""

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
    config_data_file: str, config_model_file: str, config_api_file: str, version: str
) -> None:
    config_data = build({"config": config_data_file, "version": version})
    config_model = Config.parse(config_model_file)
    config_api = Config.parse(config_api_file)

    val_data = UbuntuDataSet(
        config_data["root"] + config_data["target"],
        config_data["raw"]["val"],
        config_model["max_seq"],
        config_data["target"],
    )

    val_dataloader = UbuntuDataLoader(
        val_data,
        batch_size=config_model["batch_size"],
        shuffle=False,
        num_workers=8,
        collate_fn=collate,
    )

    model = RecoSAPL.load_from_checkpoint(config_api["model_path"], **config_model)

    trainer_params = {
        "gpus": [1],
        "num_nodes": 1,
        "distributed_backend": "ddp",
        "amp_level": "O2",
        "amp_backend": "native",
    }

    trainer = pl.Trainer(
        **trainer_params,
        max_epochs=1,
        logger=False,
        checkpoint_callback=False,
        #fast_dev_run=True
    )
    test_result = trainer.test(model, test_dataloaders=val_dataloader)
    logger.info(test_result)
    bleuS = BLEUScore()
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
    parser.add_argument("--config_api_file", default="./conf/api/ReCoSa.yml", type=str)
    parser.add_argument("--version", default="v0.0.8.1", type=str)
    args = parser.parse_args()
    main(
        args.config_data_file,
        args.config_model_file,
        args.config_api_file,
        args.version,
    )
