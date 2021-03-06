# Copyright (c) HephaestusProject
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
    This script was made by soeque1 at 24/07/20.
    To implement code for data pipeline. (e.g. custom class subclassing torch.utils.data.Dataset)
    The original paper uses the data by https://github.com/rkadlec/ubuntu-ranking-dataset-creator/blob/master/src/create_ubuntu_dataset.py
    This uses thd data by https://github.com/facebookresearch/ParlAI/blob/ea366da91c93f2cec8fe29b6d93b37ed96ed45bd/parlai/tasks/ubuntu/build.py
"""

import argparse
import os
from logging import getLogger

import src.core.build_data as build_data
from src.core.build_data import Config, DownloadableFile

logger = getLogger(__name__)


def build(opt: dict):
    logger.debug("Read Config")
    cfg = Config()
    cfg.add_dataset(opt["data_config"])

    dpath = os.path.join(cfg.dataset.root, cfg.dataset.target)
    if cfg.dataset.curl is None:
        logger.warning("DownloadableFile does not exist!; Off-line Data will be used.")
        return cfg
    else:
        resources = [DownloadableFile(**cfg.dataset.curl)]

    logger.debug("Check built")
    if not build_data.built(dpath, version_string=opt["version"]):
        logger.info("[building data: " + dpath + "]")
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in resources:
            downloadable_file.download_file(dpath)

        build_data.untar(dpath, cfg.dataset.curl.file_name)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=opt["version"])
    logger.debug("Done")
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        help="config path",
        type=str,
        default="./conf/dataset/ubuntu.yml",
    )

    args = parser.parse_args()
    build({"data_config": args.config_path, "version": None})
