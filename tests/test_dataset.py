import pytest  # noqa: E902
import torch
from torch.utils.data import DataLoader

from src.data.dataset import WMT14Dataset
from src.dataloader import WMT14DataLoader

test_langpair_exception_input = [
    # (langpair, max_length, mode)
    ("en-ko", 100, "train"),
    ("ko-ja", 200, "val"),
    ("en-ja", 150, "test"),
]


@pytest.mark.parametrize("langpair, max_length, mode", test_langpair_exception_input)
def test_langpair_exception(langpair, max_length, mode):
    with pytest.raises(NotImplementedError):
        WMT14Dataset(langpair, max_length, mode)


test_dataset_input = [
    # (langpair, max_length, mode)
    ("example", 100, "train"),
    ("example", 100, "val"),
    ("example", 100, "test"),
]


@pytest.mark.parametrize("langpair, max_length, mode", test_dataset_input)
def test_dataset_len(langpair, max_length, mode):
    ds = WMT14Dataset(langpair, max_length, mode)
    assert len(ds.source_padded_tokens) == len(ds)


@pytest.mark.parametrize("langpair, max_length, mode", test_dataset_input)
def test_dataset_getitem(langpair, max_length, mode):
    ds = WMT14Dataset(langpair, max_length, mode)
    item = ds[0]
    assert item["source"]["padded_token"].size() == item["source"]["mask"].size()
    assert item["target"]["padded_token"].size() == item["target"]["mask"].size()


@pytest.mark.parametrize("langpair, max_length, mode", test_dataset_input)
def test_dataset_encode(langpair, max_length, mode):
    ds = WMT14Dataset(langpair, max_length, mode)
    source_encode_test, target_encode_test = ds._encode()
    bos = ds.tokenizer.token_to_id("<bos>")
    eos = ds.tokenizer.token_to_id("<eos>")
    assert target_encode_test[0][0] == bos
    assert target_encode_test[0][-1] == eos
    assert isinstance(source_encode_test[0], torch.Tensor)


@pytest.mark.parametrize("langpair, max_length, mode", test_dataset_input)
def test_dataset_preprocess(langpair, max_length, mode):
    ds = WMT14Dataset(langpair, max_length, mode)
    (
        source_padded_tokens_test,
        source_masks_test,
        source_lengths_test,
        target_padded_tokens_test,
        target_masks_test,
        target_lengths_test,
    ) = ds.preprocess()
    assert source_padded_tokens_test.size() == source_masks_test.size()
    assert target_padded_tokens_test.size() == target_masks_test.size()


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
