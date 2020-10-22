import unittest
from logging import getLogger

import pytest
from fastapi.testclient import TestClient

from server import app

from .test_model import SEED_NUM

logger = getLogger(__name__)


class TestReCoSaAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        response = self.client.get("/hello")
        assert response.status_code == 200
        assert response.json() == "hi"

    def test_predict(self):
        body = {
            "input_text": "it's not out . \n they probabaly are waiting for all the mirrors to sync. the release annocement will be after that. ."
        }
        response = self.client.post("/model", json=body)
        assert response.status_code == 200
        assert response.json() == {
            "prediction": "i'm not sure, but it's not a good idea. "
        }


if __name__ == "__main__":
    unittest.main()
