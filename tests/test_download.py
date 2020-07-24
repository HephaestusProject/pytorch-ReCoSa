import unittest

from src.core.build_data import Config, DownloadableFile


class TestConifg(unittest.TestCase):
    def setUp(self):
        self.config = Config.parse("./conf/dataset/ubuntu.yml")

    def test_keys(self):
        self.assertEqual(
            list(self.config.keys()), ["root", "target", "curl", "raw", "refinement"]
        )

    def test_download_keys(self):
        down_file = DownloadableFile(**self.config["curl"])
        self.assertEqual(
            list(down_file.__dict__.keys()), ["url", "file_name", "hashcode"]
        )


if __name__ == "__main__":
    unittest.main()
