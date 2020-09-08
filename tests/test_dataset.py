import pytest  # noqa: E902
from dataset import WMT14Dataset


@pytest.mark.parametrize("langpair", ["en-de"])
@pytest.mark.parametrize(
    "source_lines",
    [
        [
            "Der Bau und die Reparatur der Autostraßen...",
            "die Mitteilungen sollen den geschäftlichen kommerziellen Charakter tragen.",
        ]
    ],
)
@pytest.mark.parametrize(
    "target_lines",
    [
        [
            "Construction and repair of highways and...",
            "An announcement must be commercial character.",
        ]
    ],
)
def test_len(langpair, source_lines, target_lines):
    ds = WMT14Dataset(langpair, source_lines, target_lines)
    assert len(ds.source_lines) == len(ds)


@pytest.mark.parametrize("langpair", ["en-de"])
@pytest.mark.parametrize(
    "source_lines",
    [
        [
            "Der Bau und die Reparatur der Autostraßen...",
            "die Mitteilungen sollen den geschäftlichen kommerziellen Charakter tragen.",
        ]
    ],
)
@pytest.mark.parametrize(
    "target_lines",
    [
        [
            "Construction and repair of highways and...",
            "An announcement must be commercial character.",
        ]
    ],
)
def test_getitem(langpair, source_lines, target_lines):
    ds = WMT14Dataset(langpair, source_lines, target_lines)
    source_encode_pad_test, target_encode_pad_test = ds[0]
    assert source_encode_pad_test.size() == target_encode_pad_test.size()
    assert source_encode_pad_test.size()[0] == ds.model_config.max_len


@pytest.mark.parametrize("langpair", ["en-de"])
@pytest.mark.parametrize(
    "source_lines",
    [
        [
            "Der Bau und die Reparatur der Autostraßen...",
            "die Mitteilungen sollen den geschäftlichen kommerziellen Charakter tragen.",
        ]
    ],
)
@pytest.mark.parametrize(
    "target_lines",
    [
        [
            "Construction and repair of highways and...",
            "An announcement must be commercial character.",
        ]
    ],
)
def test_encode(langpair, source_lines, target_lines):
    ds = WMT14Dataset(langpair, source_lines, target_lines)
    source_encode_test, target_encode_test = ds._encode(
        source_lines[0], target_lines[0]
    )
    bos = ds.tokenizer.token_to_id("<bos>")
    eos = ds.tokenizer.token_to_id("<eos>")
    assert target_encode_test[0] == bos
    assert target_encode_test[-1] == eos
    assert isinstance(source_encode_test, list)
    assert isinstance(source_encode_test[0], int)


@pytest.mark.parametrize("langpair", ["en-de"])
@pytest.mark.parametrize(
    "source_lines",
    [
        [
            "Der Bau und die Reparatur der Autostraßen...",
            "die Mitteilungen sollen den geschäftlichen kommerziellen Charakter tragen.",
        ]
    ],
)
@pytest.mark.parametrize(
    "target_lines",
    [
        [
            "Construction and repair of highways and...",
            "An announcement must be commercial character.",
        ]
    ],
)
def test_collate(langpair, source_lines, target_lines):
    ds = WMT14Dataset(langpair, source_lines, target_lines)
    source_encode_pad_test, target_encode_pad_test = ds.collate(
        source_lines[0], target_lines[0]
    )
    assert source_encode_pad_test.size() == target_encode_pad_test.size()
    assert source_encode_pad_test.size()[0] == ds.model_config.max_len
