import unittest

import pytest

from src.core.build_data import Config, DownloadableFile
from src.utils.prepare import build


class TestConifg(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.add_dataset("./conf/dataset/ubuntu.yml")

    def test_keys(self):
        self.assertEqual(
            list(self.cfg.dataset.keys()),
            ["root", "target", "curl", "raw", "refinement"],
        )

    def test_download_keys(self):
        down_file = DownloadableFile(**self.cfg.dataset.curl)
        self.assertEqual(
            list(down_file.__dict__.keys()), ["url", "file_name", "hashcode"]
        )

    def test_root(self):
        self.assertEqual(self.cfg.dataset.root, "./data/")

    def test_target(self):
        self.assertEqual(self.cfg.dataset.target, "Ubuntu")

    @pytest.mark.skip(reason="The test-data file to be changed")
    def test_download(self):
        build({"config": "./conf/dataset/ubuntu.yml", "version": "test"})


if __name__ == "__main__":
    unittest.main()
