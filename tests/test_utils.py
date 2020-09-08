import pytest  # noqa: F401
from src.utils import read_lines

test_filepath = "data/example.de"


@pytest.mark.parametrize("filepath", test_filepath)
def test_read_lines(filepath):
    de = read_lines(filepath)
    assert isinstance(de, list)
    assert (
        de[0]
        == "iron cement ist eine gebrauchs-fertige Paste, die mit einem Spachtel oder den Fingern als Hohlkehle in die Formecken (Winkel) der Stahlguss -Kokille aufgetragen wird."
    )
