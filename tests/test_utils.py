from pathlib import Path

import pytest  # noqa: F401
from omegaconf import DictConfig

from src.utils import get_config, get_configs, read_lines

test_read_lines_input = [
    # (filepath)
    Path(__file__).parent
    / "data/example.de"
]


@pytest.mark.parametrize("filepath", test_read_lines_input)
def test_read_lines(filepath):
    de = read_lines(filepath)
    assert isinstance(de, list)
    assert (
        de[0]
        == "iron cement ist eine gebrauchs-fertige Paste, die mit einem Spachtel oder den Fingern als Hohlkehle in die Formecken (Winkel) der Stahlguss -Kokille aufgetragen wird."
    )


test_get_configs_input = [
    # (langpair, args)
    ("en-de", ("data", "tokenizer", "model")),
    ("ende", ("tokenizer", "model")),
    ("EN-DE", ("tokenizer", "model")),
    ("ENDE", ("data", "tokenizer")),
]


@pytest.mark.parametrize("langpair, args", test_get_configs_input)
def test_get_configs(langpair, args):
    configs = get_configs(*args, langpair=langpair)
    assert isinstance(configs, DictConfig)
    assert len(configs) > 0


test_get_config_input = [
    # (langpair, arg)
    ("en-de", "data"),
    ("ende", "tokenizer"),
    ("EN-DE", "model"),
    ("ENDE", "data"),
]


@pytest.mark.parametrize("langpair, arg", test_get_config_input)
def test_get_config(langpair, arg):
    config = get_config(arg, langpair=langpair)
    assert isinstance(config, DictConfig)
    assert len(config) > 0


test_get_config_directory_error_input = [
    # (langpair, arg)
    ("en-de", "tokenization")
]


@pytest.mark.parametrize("langpair, arg", test_get_config_directory_error_input)
def test_get_config_directory_error(langpair, arg):
    with pytest.raises(NotADirectoryError):
        get_config(arg, langpair=langpair)
