import pytest  # noqa: E902
from torch.utils.data import DataLoader

from src.dataset import WMT14DataLoader, WMT14Dataset

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


test_dataset_input = [
    # (langpair, source_lines, target_lines)
    (
        "en-de",
        [
            "Construction and repair of highways and...",
            "This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence. This is a very very long sentence.",
        ],
        [
            "Der Bau und die Reparatur der Autostraßen...",
            "Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz. Dies ist ein sehr sehr langer Satz.",
        ],
    )
]


@pytest.mark.parametrize("langpair, source_lines, target_lines", test_dataset_input)
def test_dataset_len(langpair, source_lines, target_lines):
    ds = WMT14Dataset(langpair, source_lines, target_lines)
    assert len(ds.source_lines) == len(ds)


@pytest.mark.parametrize("langpair, source_lines, target_lines", test_dataset_input)
def test_dataset_getitem(langpair, source_lines, target_lines):
    ds = WMT14Dataset(langpair, source_lines, target_lines)
        assert source_encode_pad_test.size()[0] == ds.configs.model.model_params.max_len
        assert target_encode_pad_test.size()[0] == ds.configs.model.model_params.max_len


@pytest.mark.parametrize("langpair, source_lines, target_lines", test_dataset_input)
def test_dataset_encode(langpair, source_lines, target_lines):
    ds = WMT14Dataset(langpair, source_lines, target_lines)
    for source_line, target_line in zip(source_lines, target_lines):
        source_encode_test, target_encode_test = ds._encode(source_line, target_line)
        bos = ds.tokenizer.token_to_id("<bos>")
        eos = ds.tokenizer.token_to_id("<eos>")
        assert target_encode_test[0] == bos
        assert target_encode_test[-1] == eos
        assert isinstance(source_encode_test, list)
        assert isinstance(source_encode_test[0], int)


@pytest.mark.parametrize("langpair, source_lines, target_lines", test_dataset_input)
def test_dataset_collate(langpair, source_lines, target_lines):
    ds = WMT14Dataset(langpair, source_lines, target_lines)
    for source_line, target_line in zip(source_lines, target_lines):
        assert source_encode_pad_test.size()[0] == ds.configs.model.model_params.max_len
        assert target_encode_pad_test.size()[0] == ds.configs.model.model_params.max_len


test_dataloader_input = [
    # (langpair)
    ("example")
]


@pytest.mark.parametrize("langpair", test_dataloader_input)
def test_dataloader_setup(langpair):
    dl = WMT14DataLoader(langpair)
    dl.setup("fit")
    dl.setup("test")
    dl.setup()


@pytest.mark.parametrize("langpair", test_dataloader_input)
def test_dataloader_train_dataloader(langpair):
    dl = WMT14DataLoader(langpair)
    dl.setup("fit")
    train_dataloader = dl.train_dataloader()
    val_dataloader = dl.valid_dataloader()
    assert isinstance(train_dataloader, DataLoader)
    assert isinstance(val_dataloader, DataLoader)


@pytest.mark.parametrize("langpair", test_dataloader_input)
def test_dataloader_test_dataloader(langpair):
    dl = WMT14DataLoader(langpair)
    dl.setup("test")
    test_dataloader = dl.test_dataloader()
    assert isinstance(test_dataloader, DataLoader)
