# template

[![Code Coverage](https://codecov.io/gh/HephaestusProject/pytorch-ReCoSa/branch/master/graph/badge.svg)](https://codecov.io/gh/HephaestusProject/pytorch-ReCoSa)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Abstract

* ReCoSA is able to detect relevant contexts and produce a suitable response accordingly. Firstly, a word level LSTM encoder is conducted to obtain the initial representation of each context. Then, the self-attention mechanism is utilized to update both the context and masked response representation. Finally, the attention weights between each context and response representations are computed and used in the further decoding process.

## Table

* DSTC7_AVSD

| PPL   |      BLEU(4-grams)    | BLEU(2-grams)      |
|----------|:-------------:|:-------------:|
| 124.72 |  0.106 | 0.223

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

* interface
  + ArgumentParser의 command가 code block 형태로 들어가야함.
    - single-gpu, multi-gpu

### Inference

* interface
  + ArgumentParser의 command가 code block 형태로 들어가야함.

### Evaluate

```sh
./evaluate.sh
```

### Project structure

```
.
├── apply.sh
├── conf
│   ├── dataset
│   │   └── ubuntu.yml
│   └── model
│       └── your_model.yml
├── data
│   └── Ubuntu
│       ├── LICENSE
│       ├── test.csv
│       ├── train.csv
│       └── valid.csv
├── evaluate.py
├── infer.py
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
│   ├── __init__.py
│   └── test_download.py
└── train.py
```

### License

* Licensed under an MIT license.
