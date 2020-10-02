from typing import List, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import WMT14Dataset
from src.utils import get_configs


class WMT14DataLoader(LightningDataModule):
    """Load WMT14 dataset and prepare batches for training and testing transformer.

    Attributes:
        langpair: language pair to translate
    """

    def __init__(self, langpair: str) -> None:
        super().__init__()
        self.configs = get_configs("data", "model", langpair=langpair)
        self.langpair = langpair
        self.max_length = self.configs.model.model_params.max_len

    def setup(self, stage: Optional[str] = None) -> None:
        """Assign dataset for use in dataloaders

        Args:
            stage: decide to load train/val or test
        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = WMT14Dataset(
                self.langpair, max_length=self.max_length, mode="train"
            )
            self.valid_dataset = WMT14Dataset(
                self.langpair, max_length=self.max_length, mode="val"
            )
        # Assign test dataset for use in dataloaders
        if stage == "test" or stage is None:
            self.test_dataset = WMT14Dataset(
                self.langpair, max_length=self.max_length, mode="test"
            )

    def batch_by_tokens(
        self, dataset: Dataset, max_tokens: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Create mini-batch tensors by number of tokens

        Args:
            dataset: source and target dataset containing padded_token, mask, and length
                     e.g.,
                     {'source': {'padded_token': torch.Tensor, 'mask': torch.Tensor, 'length': torch.Tensor},
                      'target': {'padded_token': torch.Tensor, 'mask': torch.Tensor, 'length': torch.Tensor}}
            max_tokens: max number of tokens per batch

        Returns:
            indices_batches:
        """
        max_tokens = (
            25000 if max_tokens is None else self.configs.model.train_hparams.batch_size
        )

        start_idx = 0
        source_sample_lens, target_sample_lens = [], []
        indices_batches = []
        for end_idx in range(len(dataset)):
            source_sample_lens.append(dataset[end_idx]["source"]["length"])
            target_sample_lens.append(dataset[end_idx]["target"]["length"])
            # when batch is full
            if (
                sum(source_sample_lens) > max_tokens
                or sum(target_sample_lens) > max_tokens
            ):
                indices_batch = torch.arange(start_idx, end_idx)
                indices_batches.append(indices_batch)
                start_idx = end_idx
                source_sample_lens, target_sample_lens = [source_sample_lens[-1]], [
                    target_sample_lens[-1]
                ]  # end_idx is not included
            # when iteration ends
            elif end_idx == len(dataset):
                indices_batch = torch.arange(start_idx, end_idx)
                indices_batches.append(indices_batch)
        return indices_batches

    # TODO: batch together by approx. sequence length.
    def train_dataloader(self) -> DataLoader:
        batch_sampler = self.batch_by_tokens(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            shuffle=False,
            drop_last=False,
            num_workers=self.configs.model.data_params.num_workers,
        )

    def valid_dataloader(self) -> DataLoader:
        batch_sampler = self.batch_by_tokens(self.valid_dataset)
        return DataLoader(
            self.valid_dataset,
            batch_sampler=batch_sampler,
            shuffle=False,
            drop_last=False,
            num_workers=self.configs.model.data_params.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        batch_sampler = self.batch_by_tokens(self.test_dataset)
        return DataLoader(
            self.test_dataset,
            batch_sampler=batch_sampler,
            shuffle=False,
            drop_last=False,
            num_workers=self.configs.model.data_params.num_workers,
        )
