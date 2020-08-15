import pytest  # noqa: F401

import tokenizers
from tokenizer.encode import load_tokenizer, encode, encode_batch, tokenize, tokenize_batch

tokenizer = load_tokenizer()


def test_encode():
    line = "Hello"
    result = encode(tokenizer, line)
    assert type(result) == tokenizers.Encoding
    assert result.tokens == ["▁He", "llo"]


def test_encode_batch():
    lines = ["Hello", "Jihyung"]
    results = encode_batch(tokenizer, lines)
    assert type(results) == list
    assert results[0].tokens == ["▁He", "llo"]
    assert results[1].tokens == ["▁J", "ih", "y", "ung"]


def test_tokenize():
    line = "Hello"
    result = tokenize(tokenizer, line)
    assert result == ["▁He", "llo"]


def test_tokenize_batch():
    lines = ["Hello", "Jihyung"]
    results = tokenize_batch(tokenizer, lines)
    assert results == [["▁He", "llo"], ["▁J", "ih", "y", "ung"]]
