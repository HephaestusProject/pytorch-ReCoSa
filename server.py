from pathlib import Path

from pydantic import BaseModel

from infer import ReCoSaAPI
from serving.app_factory import create_app
from src.core.build_data import Config

config = Config()
config.add_model("./conf/model/ReCoSa.yml")
config.add_api("./conf/api/ReCoSa.yml")

predictor = ReCoSaAPI(config)


class Request(BaseModel):
    input_text: str


class Response(BaseModel):
    prediction: str


def handler(request: Request) -> Response:
    prediction = predictor.generate(request.input_text)
    return Response(prediction=prediction)


app = create_app(handler, Request, Response)