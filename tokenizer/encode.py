from tokenizers import Tokenizer, SentencePieceBPETokenizer
from typing import List

from utils import root_dir, tokenizer_config


def load_tokenizer():
    tokenizer = SentencePieceBPETokenizer(
        f"{root_dir}/tokenizer/{tokenizer_config.tokenizer_name}-vocab.json",
        f"{root_dir}/tokenizer/{tokenizer_config.tokenizer_name}-merges.txt",
    )
    return tokenizer


def encode(tokenizer: Tokenizer, line: str):
    """Encode a single string line.

    Args:
        tokenizer: trained huggingface tokenizer
        line: raw string

    Returns:
        encoded_line: encode string

    Example:
    >>> tokenizer = load_tokenizer()
    >>> line = "Hello"
    >>> encode(tokenizer, line)
    Encoding(num_tokens=2, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])
    """
    return tokenizer.encode(line)


def encode_batch(tokenizer: Tokenizer, lines: List[str]):
    """Encode multiple string lines.

    Args:
        tokenizer: trained huggingface tokenizer
        lines: list of raw string

    Returns:
        encoded_lines: list of encoded string

    Example:
    >>> tokenizer = load_tokenizer()
    >>> lines = ['Hello', 'Jihyung']
    >>> encode_batch(tokenizer, lines)
    [Encoding(num_tokens=2, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing]),
     Encoding(num_tokens=4, attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])]
    """
    return tokenizer.encode_batch(lines)


def tokenize(tokenizer: Tokenizer, line: str):
    """Tokenize a single string line.

    Args:
        tokenizer: trained huggingface tokenizer
        line: raw string

    Returns:
        tokenized: list of tokens

    Example:
    >>> tokenizer = load_tokenizer()
    >>> line = 'Hello"
    >>> tokenize(tokenizer, line)
    ['▁He', 'llo']
    """
    return tokenizer.encode(line).tokens


def tokenize_batch(tokenizer: Tokenizer, lines: List[str]):
    """Tokenize multiple string lines.

    Args:
        tokenizer: trained huggingface tokenizer
        lines: list of raw string

    Returns:
        tokenized: list of list of tokens

    Example:
    >>> tokenizer = load_tokenizer()
    >>> lines = ['Hello', 'Jihyung']
    >>> tokenizee_batch(tokenizer, lines)
    [['▁He', 'llo'], ['▁J', 'ih', 'y', 'ung']]
    """
    encoded_results = tokenizer.encode_batch(lines)
    # TODO: multiprocessing
    tokenized = []
    for encoded_result in encoded_results:
        tokenized.append(encoded_result.tokens)
    return tokenized
