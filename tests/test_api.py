import unittest
from logging import getLogger

import pytest
from fastapi.testclient import TestClient
from .test_model import SEED_NUM

from server import app

logger = getLogger(__name__)


class TestReCoSaAPI(unittest.TestCase):
    def test_health(self):
        client = TestClient(app)
        response = client.get("/hello")
        assert response.status_code == 200
        assert response.json() == "hi"

    def test_predict(self):
        body = {
        "input_text": "it's not out . \n they probabaly are waiting for all the mirrors to sync. the release annocement will be after that. ."
        }
        client = TestClient(app)
        response = client.post(
            "/model",
            json=body
        )
        assert response.status_code == 200
        assert response.json() == {'prediction': "yeah, but I'm not sure if it's a good idea. "}


if __name__ == "__main__":
    unittest.main()
