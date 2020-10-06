from pathlib import Path

import pytest  # noqa: F401
from omegaconf import DictConfig

from src.utils import Config, read_lines

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


test_data_tokenizer_config_input = [
    # (langpair)
    ("en-de"),
    ("ENDE"),
]


@pytest.mark.parametrize("langpair", test_data_tokenizer_config_input)
def test_data_tokenizer_config(langpair):
    config = Config()
    config.add_data(langpair)
    config.add_tokenizer(langpair)
    assert type(config.data) == DictConfig
    assert type(config.tokenizer) == DictConfig
