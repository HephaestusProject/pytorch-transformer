import pytest  # noqa: E902

from src.dataset import WMT14Dataset

test_langpair_exception_input = [
    # (langpair, source_lines, target_lines)
    ("en-ko", "Hello", "안녕하세요"),
    ("ko-ja", "감사합니다", "ありがとうございました"),
]


@pytest.mark.parametrize(
    "langpair, source_lines, target_lines", test_langpair_exception_input
)
def test_langpair_exception(langpair, source_lines, target_lines):
    with pytest.raises(NotImplementedError):
        WMT14Dataset(langpair, source_lines, target_lines)


test_input = [
    # (langpair, source_lines, target_lines)
    (
        "de-en",
        [
            "Der Bau und die Reparatur der Autostraßen...",
            "Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz.",
        ],
        [
            "Construction and repair of highways and...",
            "This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence.",
        ],
    )
]


@pytest.mark.parametrize("langpair, source_lines, target_lines", test_input)
def test_len(langpair, source_lines, target_lines):
    ds = WMT14Dataset(langpair, source_lines, target_lines)
    assert len(ds.source_lines) == len(ds)


@pytest.mark.parametrize("langpair, source_lines, target_lines", test_input)
def test_getitem(langpair, source_lines, target_lines):
    ds = WMT14Dataset(langpair, source_lines, target_lines)
    for source_encode_pad_test, target_encode_pad_test in ds:
        assert source_encode_pad_test.size() == target_encode_pad_test.size()
        assert source_encode_pad_test.size()[0] == ds.configs.model.max_len


@pytest.mark.parametrize("langpair, source_lines, target_lines", test_input)
def test_encode(langpair, source_lines, target_lines):
    ds = WMT14Dataset(langpair, source_lines, target_lines)
    for source_line, target_line in zip(source_lines, target_lines):
        source_encode_test, target_encode_test = ds._encode(source_line, target_line)
        bos = ds.tokenizer.token_to_id("<bos>")
        eos = ds.tokenizer.token_to_id("<eos>")
        assert target_encode_test[0] == bos
        assert target_encode_test[-1] == eos
        assert isinstance(source_encode_test, list)
        assert isinstance(source_encode_test[0], int)


@pytest.mark.parametrize("langpair, source_lines, target_lines", test_input)
def test_collate(langpair, source_lines, target_lines):
    ds = WMT14Dataset(langpair, source_lines, target_lines)
    for source_line, target_line in zip(source_lines, target_lines):
        source_encode_pad_test, target_encode_pad_test = ds.collate(
            source_line, target_line
        )
        assert source_encode_pad_test.size() == target_encode_pad_test.size()
        assert source_encode_pad_test.size()[0] == ds.configs.model.max_len
