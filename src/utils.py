from typing import List


def read_lines(filepath: str) -> List[str]:
    """Read text file

    Args:
        filepath: path of the test file where each line is split by '\n'
    Returns:
        lines: list of lines
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    return lines
