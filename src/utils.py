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


def get_configs(langpair: str, *args: str) -> DictConfig:
    """Get all configurations regarding model training

    Args:
        langpair: language pair to train transformer
        args: configuration types (e.g., tokenizer, dataset, model)
    Returns:
        configs: a single configuration that merged dataset, tokenizer, and model configurations
    """
    langpair = normalize_langpair(langpair)
    configs = OmegaConf.create()

    for arg in args:
        config = get_config(langpair, arg)
        configs[arg] = config

    OmegaConf.set_readonly(configs, True)
    return configs


def get_config(langpair: str, arg: str) -> DictConfig:
    """Get a configuration designated by arg

    Args:
        langpair: language pair
        arg: configuration type
    Returns:
        config: configuration related to the arg
    """
    langpair = normalize_langpair(langpair)
    root_dir = Path(__file__).parent.parent
    config_dir = root_dir / "configs" / arg
    if not config_dir.is_dir():
        raise NotADirectoryError(
            f"{config_dir} does not exists. Check if configuration saved directory exists."
        )

    if arg == "model":
        config_path = config_dir / "transformers.yaml"
    else:
        config_path = list(config_dir.glob(f"*{langpair}*"))[0]
        if not config_path.exists():
            raise FileNotFoundError(
                f"{config_path} does not exists. Check if configuration yaml file exists."
            )

    config = OmegaConf.load(config_path)

    if arg == "tokenizer":
        config.tokenizer_vocab = str(
            root_dir / "tokenizer" / (config.tokenizer_name + "-vocab.json")
        )
        config.tokenizer_merges = str(
            root_dir / "tokenizer" / (config.tokenizer_name + "-merges.txt")
        )

    return config
