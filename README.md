# template

[![Code Coverage](https://codecov.io/gh/HephaestusProject/pytorch-ReCoSa/branch/master/graph/badge.svg)](https://codecov.io/gh/HephaestusProject/pytorch-ReCoSa)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Abstract

* ReCoSA is able to detect relevant contexts and produce a suitable response accordingly. Firstly, a word level LSTM encoder is conducted to obtain the initial representation of each context. Then, the self-attention mechanism is utilized to update both the context and masked response representation. Finally, the attention weights between each context and response representations are computed and used in the further decoding process.

## Table (v0.2.3)

* DSTC7_AVSD

| PPL   |      BLEU(4-grams)    | BLEU(2-grams)      |
|----------|:-------------:|:-------------:|
| 105.34 |  0.098 | 0.214

## Training history

* tensorboard 또는 weights & biases를 이용, 학습의 로그의 스크린샷을 올려주세요.

## OpenAPI로 Inference 하는 방법

```sh
  curl -s "http://127.0.0.1:8000/hello"
  curl -X POST "http://127.0.0.1:8000/model" -H "Content-Type: application/json" -d "{\"input_text\":\"thanks! \"}"
```

## Usage

### Environment

* install from source code
* dockerfile 이용

### Training & Evaluate

```sh
./train.sh
```

```sh
./evaluate.sh
```

### Project structure

```sh
.
├── Dockerfile
├── LICENSE
├── README.md
├── apply.sh
|── conf
│   ├── api
│   │   ├── ReCoSa.yml
│   │   └── ReCoSa_AVSD.yml
│   ├── dataset
│   │   ├── DSTC7_AVSD.yml
│   │   ├── ubuntu.yml
│   │   └── ubuntu_test.yml
│   ├── model
│   │   ├── ReCoSa.yml
│   │   └── ReCoSa_test.yml
│   ├── tokenizer
│   │   ├── added_tokens.json
│   │   ├── merges.txt
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   └── trainer
│       ├── ReCoSa.yml
│       └── ReCoSa_test.yml
├── configs
│   └── deploying
│       └── latest.yaml
├── coverage.xml
├── data
│   ├── DSTC7_AVSD
│   │   ├── dial.dev
│   │   ├── dial.test
│   │   └── dial.train
│   ├── DailyDialog
│   │   ├── dial.test
│   │   ├── dial.train
│   │   └── dial.valid
│   ├── PersonaChat
│   │   ├── dial.test
│   │   ├── dial.train
│   │   └── dial.valid
│   └── Ubuntu
│       ├── LICENSE
│       ├── test.csv
│       ├── train.csv
│       └── valid.csv
├── deploying
│   └── helm
│       ├── Chart.yaml
│       ├── templates
│       │   ├── deployment.yaml
│       │   └── service.yaml
│       └── values.yaml
├── evaluate.py
├── evaluate.sh
├── infer.py
├── lightning_logs
├── requirements.txt
├── server.Dockerfile
├── serving
│   ├── __init__.py
│   └── app_factory.py
├── serving
├── LICENSE
├── README.md
├── requirements.txt
├── src
│   ├── core
│   │   └── build_data.py
│   ├── data.py
│   ├── metric.py
│   ├── model
│   │   ├── net.py
│   │   └── ops.py
│   ├── utils
│   │   └── prepare.py
│   └── utils.py
├── tests
│   ├── resources
│   │   └── Ubuntu
│   │       └── sample.csv
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_download.py
│   ├── test_inference.py
│   ├── test_model.py
│   └── test_trainer.py
├── train.py
└── train.sh
```

### License

* Licensed under an MIT license.
