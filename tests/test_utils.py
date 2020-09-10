from pathlib import Path

import pytest  # noqa: F401
from omegaconf import DictConfig

from src.utils import read_lines, get_configs

test_read_lines_input = [
    # (filepath)
    Path(__file__).parent / "data/example.de"
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
    # (langpair)
    "de-en",
    "deen"
]


@pytest.mark.parametrize("langpair", test_get_configs_input)
def test_get_configs(langpair):
    configs = get_configs(langpair)
    assert isinstance(configs.dataset, DictConfig)
    assert isinstance(configs.tokenizer, DictConfig)
    assert isinstance(configs.model, DictConfig)

    assert len(configs.dataset) > 0
    assert len(configs.tokenizer) > 0
    assert len(configs.model) > 0
