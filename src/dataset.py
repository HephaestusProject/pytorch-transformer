from typing import List, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from tokenizers import SentencePieceBPETokenizer
from torch.utils.data import DataLoader, Dataset

from .utils import get_configs, read_lines


class WMT14Dataset(Dataset):
    """Dataset to train and test transformer

    Attributes:
        langpair: language pair to translate
        source_lines: list of text in source language
        target_lines: list of text in target language
    """

    def __init__(
        self, langpair: str, source_lines: List[str], target_lines: List[str]
    ) -> None:
        super().__init__()
        self.configs = get_configs(langpair)
        self.tokenizer = self.load_tokenizer()
        self.source_lines = source_lines
        self.target_lines = target_lines

    def __len__(self) -> int:
        return len(self.source_lines)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        source_encoded, target_encoded = self.collate(
            self.source_lines[index], self.target_lines[index]
        )
        return source_encoded, target_encoded

    def _encode(
        self, source_line: str, target_line: str
    ) -> Tuple[List[int], List[int]]:
        """Encode string line to index

        Args:
            source_line: raw text in source language
            target_line: raw text in target language

        Returns:
            source_encoded: encoded ids of source_line
            target_encoded: encoded ids of target_line. Unlike source_encoded, <bos> and <eos> are added.
        """
        source_encoded = self.tokenizer.encode(source_line).ids
        target_encoded = self.tokenizer.encode(target_line).ids
        bos = self.tokenizer.token_to_id("<bos>")
        eos = self.tokenizer.token_to_id("<eos>")
        target_encoded.insert(0, bos)
        target_encoded.append(eos)
        return source_encoded, target_encoded

    def collate(
        self, source_line: str, target_line: str
    ) -> Tuple[
        torch.Tensor, torch.Tensor
    ]:  # TODO: try to be more efficient (batch-level collate)
        """Collate source and target text

        Args:
            source_line: raw text in source language
            target_line: raw text in target language

        Returns:
            source_encoded: padded encodings of source_line
            target_encoded: padded encodings of target_line
        """
        max_len = self.configs.model.max_len
        pad = self.tokenizer.token_to_id("<pad>")

        source_encoded, target_encoded = self._encode(source_line, target_line)
        if len(source_encoded) >= max_len:
            source_encoded = source_encoded[:max_len]
        else:
            pad_length = max_len - len(source_encoded)
            source_encoded = source_encoded + [pad] * pad_length

        if len(target_encoded) >= max_len:
            target_encoded = target_encoded[:max_len]
        else:
            pad_length = max_len - len(target_encoded)
            target_encoded = target_encoded + [pad] * pad_length

        assert len(source_encoded) == len(target_encoded)
        source_encoded, target_encoded = (
            torch.tensor(source_encoded),
            torch.tensor(target_encoded),
        )
        return source_encoded, target_encoded

    def load_tokenizer(self):
        tokenizer = SentencePieceBPETokenizer(
            vocab_file=self.configs.tokenizer.tokenizer_vocab,
            merges_file=self.configs.tokenizer.tokenizer_merges,
        )
        return tokenizer


class WMT14DataLoader(LightningDataModule):
    """Load WMT14 dataset to train and test transformer

    Attributes:
        langpair: language pair to translate
    """

    def __init__(self, langpair: str) -> None:
        super().__init__()
        self.configs = get_configs(langpair)
        self.langpair = langpair

    def setup(self, stage: Optional[str] = None) -> None:
        """Assign dataset for use in dataloaders

        Args:
            stage: decide to load train/val or test
        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.source_train = read_lines(self.configs.dataset.path.source_train)
            self.source_val = read_lines(self.configs.dataset.path.source_dev)
            self.target_train = read_lines(self.configs.dataset.path.target_train)
            self.target_val = read_lines(self.configs.dataset.path.target_dev)
            assert len(self.source_train) == len(self.target_train)
            assert len(self.source_val) == len(self.target_val)
        # Assign test dataset for use in dataloaders
        if stage == "test" or stage is None:
            self.source_test = read_lines(self.configs.dataset.path.source_test)
            self.target_test = read_lines(self.configs.dataset.path.target_test)
            assert len(self.source_test) == len(self.target_test)

    def train_dataloader(self) -> DataLoader:
        train_dataset = WMT14Dataset(
            self.langpair, self.source_train, self.target_train
        )
        return DataLoader(
            train_dataset,
            batch_size=self.configs.model.train_hparams.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.configs.model.data_params.num_workers,
        )

    def valid_dataloader(self) -> DataLoader:
        val_dataset = WMT14Dataset(self.langpair, self.source_val, self.target_val)
        return DataLoader(
            val_dataset,
            batch_size=self.configs.model.train_hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.configs.model.data_params.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        test_dataset = WMT14Dataset(self.langpair, self.source_test, self.target_test)
        return DataLoader(
            test_dataset,
            batch_size=self.configs.model.train_hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.configs.model.data_params.num_workers,
        )
