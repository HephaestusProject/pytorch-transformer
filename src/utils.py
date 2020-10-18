from pathlib import Path, PosixPath
from typing import List

import torch
from omegaconf import DictConfig, OmegaConf
from tokenizers import SentencePieceBPETokenizer


def read_lines(filepath: str) -> List[str]:
    """Read text file

    Args:
        filepath: path of the test file where each line is split by '\n'
    Returns:
        lines: list of lines
    """
    filepath = Path(filepath)
    with filepath.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    return lines


def normalize_langpair(langpair: str) -> str:
    """Normalize language pair

    Args:
        langpair: language pairs in various formats
    Returns:
        langpair_norm: language pair in normalized format
    """
    if langpair in ["en-de", "ende", "ENDE", "EN-DE"]:
        langpair_norm = "en-de"
    # TODO: add en-fr
    #  elif langpair in ["en-fr", "enfr", "EN-FR", "ENFR"]:
    #  langpair_norm = "en-fr"
    elif langpair == "example":  # for test
        langpair_norm = langpair
    else:
        raise NotImplementedError(
            f'{langpair} is not supported, since Hephaestus project aims to reproduce "Attention is all you need".'
        )
    return langpair_norm


def load_tokenizer(langpair: str) -> SentencePieceBPETokenizer:
    if langpair in ["en-de", "de-en", "ende", "deen", "ENDE", "EN-DE"]:
        langpair = "deen"

    tokenizer_dir = Path(__file__).parent.parent / "src" / "tokenizer"
    vocab_filepath = (
        tokenizer_dir / f"sentencepiece_bpe_wmt14_{langpair}.tokenizer-vocab.json"
    )
    merges_filepath = (
        tokenizer_dir / f"sentencepiece_bpe_wmt14_{langpair}.tokenizer-merges.txt"
    )

    tokenizer = SentencePieceBPETokenizer(
        vocab=str(vocab_filepath),
        merges=str(merges_filepath),
    )
    return tokenizer


class Config:
    """Load configuration files via OmegaConf"""

    def __init__(self) -> None:
        root_dir = Path(__file__).parent.parent
        self.config_dir: PosixPath = root_dir / "configs"
        self.configs: DictConfig = OmegaConf.create()

    def add_data(self, langpair: str) -> None:
        langpair = normalize_langpair(langpair)
        data_config = self.config_dir / "data" / f"wmt14.{langpair}.yaml"
        self.configs.update({"data": OmegaConf.load(data_config)})

    def add_tokenizer(self, langpair: str) -> None:
        langpair = normalize_langpair(langpair)
        tokenizer_config = (
            self.config_dir / "tokenizer" / f"sentencepiece_bpe_wmt14_{langpair}.yaml"
        )
        self.configs.update({"tokenizer": OmegaConf.load(tokenizer_config)})

    def add_model(self, is_base: bool = True) -> None:
        model_type = "base" if is_base else "big"
        model_config = self.config_dir / "model" / f"transformer-{model_type}.yaml"
        self.configs.update({"model": OmegaConf.load(model_config)})

    @property
    def data(self) -> DictConfig:
        return self.configs.data

    @property
    def tokenizer(self) -> DictConfig:
        return self.configs.tokenizer

    @property
    def model(self) -> DictConfig:
        return self.configs.model
