import pytest
from src.load_dataset import WMT14DataModule


@pytest.mark.parametrize("langpair", ["de-en"])
def test_setup(langpair):
    dm = WMT14DataModule(langpair)
