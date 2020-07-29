# template

[![Code Coverage](https://codecov.io/gh/HephaestusProject/template/branch/master/graph/badge.svg)](https://codecov.io/gh/HephaestusProject/template)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring signiﬁcantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.


## Table

* 구현하는 paper에서 제시하는 benchmark dataset을 활용하여 구현하여, 논문에서 제시한 성능과 비교합니다.
  + benchmark dataset은 하나만 골라주세요.
    1. 논문에서 제시한 hyper-parameter와 architecture로 재현을 합니다.
    2. 만약 재현이 안된다면, 본인이 변경한 사항을 서술해주세요.

## Training history

* tensorboard 또는 weights & biases를 이용, 학습의 로그의 스크린샷을 올려주세요.

## OpenAPI로 Inference 하는 방법

* curl ~~~

## Usage

### Environment

* install from source code
* dockerfile 이용

### Tokenize

This repository assumes that tokenizer is already prepared.
Below is a detailed description of tokenizer training.

- tokenizer config: `configs/tokenizer/`

```bash
python train_bpe.py
```

### Training & Evaluate

* interface
  + ArgumentParser의 command가 code block 형태로 들어가야함.
    - single-gpu, multi-gpu

### Inference

* interface
  + ArgumentParser의 command가 code block 형태로 들어가야함.

### Project structure

* 터미널에서 tree커맨드 찍어서 붙이세요.

### License

* Licensed under an MIT license.
