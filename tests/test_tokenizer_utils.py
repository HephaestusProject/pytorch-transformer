import pytest  # noqa: F401

from tokenizer.utils import read_lines, root_dir


def test_read_lines():
    example_de_path = f"{root_dir}/dataset/example.de"
    de = read_lines(example_de_path)
    assert len(de) == 10
    assert (
        de[0]
        == "iron cement ist eine gebrauchs-fertige Paste, die mit einem Spachtel oder den Fingern als Hohlkehle in die Formecken (Winkel) der Stahlguss -Kokille aufgetragen wird."
    )
