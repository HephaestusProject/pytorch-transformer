from pathlib import Path
from typing import List

from omegaconf import DictConfig, OmegaConf


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


def get_configs(langpair: str) -> DictConfig:
    """Get all configurations regarding model training

    Args:
        langpair: language pair to train transformer
    Returns:
        configs: a single configuration that merged dataset, tokenizer, and model configurations
    """
    langpair = normalize_langpair(langpair)

    root_dir = Path(__file__).parent.parent
    dataset_config_dir = root_dir / "configs" / "dataset"
    tokenizer_config_dir = root_dir / "configs" / "tokenizer"
    model_config_dir = root_dir / "configs" / "model"

    configs = OmegaConf.create()

    model_config_path = model_config_dir / "transformers.yaml"
    model_config = OmegaConf.load(model_config_path)

    dataset_config_path = dataset_config_dir / f"wmt14.{langpair}.yaml"
    tokenizer_config_path = (
        tokenizer_config_dir / f"sentencepiece_bpe_wmt14_{langpair}.yaml"
    )
    dataset_config = OmegaConf.load(dataset_config_path)
    tokenizer_config = OmegaConf.load(tokenizer_config_path)
    tokenizer_config.tokenizer_vocab = str(
        root_dir / "tokenizer" / (tokenizer_config.tokenizer_name + "-vocab.json")
    )
    tokenizer_config.tokenizer_merges = str(
        root_dir / "tokenizer" / (tokenizer_config.tokenizer_name + "-merges.txt")
    )

    configs.update(
        dataset=dataset_config, tokenizer=tokenizer_config, model=model_config
    )
    return configs
