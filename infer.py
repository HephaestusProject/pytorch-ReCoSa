"""
    This script was made by soeque1 at 24/07/20.
    To implement code for inference with your model.
"""

import torch

from src.model.net import ReCoSa


class ReCoSaAPI:
    def __init__(
        self, model_config: dict, _device: torch.device, api_conifg: dict
    ) -> None:
        self.model = ReCoSa(model_config, _device)
        self.model.eval()
        self.api_conifg = api_config

    def predict(self, _input: dict) -> dict:
        _ctx = _input["text"]
        response = self.model.predict(_ctx)
        out = {"response": response}
        return out


if __name__ == "__main__":
    pass
