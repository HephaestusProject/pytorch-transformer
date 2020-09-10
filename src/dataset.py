from typing import List, Tuple, Optional

import torch
from tokenizers import SentencePieceBPETokenizer
from torch.utils.data import Dataset

from .utils import get_configs, read_lines


class WMT14Dataset(Dataset):
    """Dataset to train and test transformer

    Attributes:
        langpair: language pair to translate
        source_lines: list of text in source language
        target_lines: list of text in target language
    """

    def __init__(
        self, langpair: str, source_lines: List[str], target_lines: List[str]
    ) -> None:
        super().__init__()
        self.configs = get_configs(langpair)
        self.tokenizer = self.load_tokenizer()
        self.source_lines = source_lines
        self.target_lines = target_lines

    def __len__(self) -> int:
        return len(self.source_lines)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        source_encoded, target_encoded = self.collate(
            self.source_lines[index], self.target_lines[index]
        )
        return source_encoded, target_encoded

    def _encode(
        self, source_line: str, target_line: str
    ) -> Tuple[List[int], List[int]]:
        """Encode string line to index

        Args:
            source_line: raw text in source language
            target_line: raw text in target language

        Returns:
            source_encoded: encoded ids of source_line
            target_encoded: encoded ids of target_line. Unlike source_encoded, <bos> and <eos> are added.
        """
        source_encoded = self.tokenizer.encode(source_line).ids
        target_encoded = self.tokenizer.encode(target_line).ids
        bos = self.tokenizer.token_to_id("<bos>")
        eos = self.tokenizer.token_to_id("<eos>")
        target_encoded.insert(0, bos)
        target_encoded.append(eos)
        return source_encoded, target_encoded

    def collate(
        self, source_line: str, target_line: str
    ) -> Tuple[
        torch.Tensor, torch.Tensor
    ]:  # TODO: try to be more efficient (batch-level collate)
        """Collate source and target text

        Args:
            source_line: raw text in source language
            target_line: raw text in target language

        Returns:
            source_encoded: padded encodings of source_line
            target_encoded: padded encodings of target_line
        """
        max_len = self.configs.model.max_len
        pad = self.tokenizer.token_to_id("<pad>")

        source_encoded, target_encoded = self._encode(source_line, target_line)
        if len(source_encoded) >= max_len:
            source_encoded = source_encoded[:max_len]
        else:
            pad_length = max_len - len(source_encoded)
            source_encoded = source_encoded + [pad] * pad_length

        if len(target_encoded) >= max_len:
            target_encoded = target_encoded[:max_len]
        else:
            pad_length = max_len - len(target_encoded)
            target_encoded = target_encoded + [pad] * pad_length

        assert len(source_encoded) == len(target_encoded)
        source_encoded, target_encoded = (
            torch.tensor(source_encoded),
            torch.tensor(target_encoded),
        )
        return source_encoded, target_encoded
