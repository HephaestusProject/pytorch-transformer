import pytest  # noqa: F401

from tokenizer.encode import load_tokenizer, tokenize, tokenize_batch

tokenizer = load_tokenizer()


def test_tokenize():
    line = 'Hello'
    result = tokenize(tokenizer, line)
    assert result == ['▁He', 'llo']


def test_tokenize_batch():
    lines = ['Hello', 'Jihyung']
    results = tokenize_batch(tokenizer, lines)
    assert results == [['▁He', 'llo'], ['▁J', 'ih', 'y', 'ung']]
