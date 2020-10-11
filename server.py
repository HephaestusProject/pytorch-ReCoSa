from pathlib import Path

from pydantic import BaseModel

from serving.app_factory import create_app
from src.core.build_data import Config
from infer import Predictor
from train import RecoSAPL

config = Config()
config.add_model("./conf/model/ReCoSa.yml")
config.add_api("./conf/api/ReCoSa.yml")

predictor = Predictor.from_checkpoint(RecoSAPL, config)


class Request(BaseModel):
    input_text: str


class Response(BaseModel):
    prediction: str


def handler(request: Request) -> Response:
    prediction = predictor.generate(request.input_text)
    return Response(prediction=prediction)


app = create_app(handler, Request, Response)
