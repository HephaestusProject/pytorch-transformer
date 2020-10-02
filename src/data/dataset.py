from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset

from src.utils import get_configs, load_tokenizer, read_lines
from src.data.data_utils import filter_by_length, pad_and_mask_sequence


class WMT14Dataset(Dataset):
    """A dataset that provides helpers for batching, such as encoding.

    Attributes:
        langpair: language pair to translate
        mode: purpose of the dataset (e.g., train, val/dev, test)
    """

    def __init__(
        self, langpair: str, max_length: int, mode: str
    ) -> None:
        super().__init__()
        root_dir = Path(__file__).parents[2]
        self.configs = get_configs("data", "tokenizer", langpair=langpair)
        self.tokenizer = load_tokenizer(self.configs.tokenizer)
        self.max_length = max_length
        self.data_config = self.configs.data  # TODO: inference
        if mode == "train":
            self.source_lines = read_lines(root_dir / self.data_config.path.source_train)
            self.target_lines = read_lines(root_dir / self.data_config.path.target_train)
        elif mode in ["val", "dev"]:
            self.source_lines = read_lines(root_dir / self.data_config.path.source_dev)
            self.target_lines = read_lines(root_dir / self.data_config.path.target_dev)
        elif mode == "test":
            self.source_lines = read_lines(root_dir / self.data_config.path.source_test)
            self.target_lines = read_lines(root_dir / self.data_config.path.target_test)
        else:
            raise ValueError("Invalid input for a dataset model. Should be one of train, val/dev, and test.")
        assert len(self.source_lines) == len(self.target_lines)
        self.source_padded_tokens, self.source_masks, self.source_lengths, self.target_padded_tokens, self.target_masks, self.target_lengths = self.preprocess()

    def __len__(self) -> int:
        return len(self.source_padded_tokens)

    def __getitem__(
        self, index: int
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        return {'source': {'padded_token': self.source_padded_tokens[index], 'mask': self.source_masks[index], 'length': self.source_lengths[index]}, 'target': {'padded_token': self.target_padded_tokens[index], 'mask': self.target_masks[index], 'length': self.target_lengths[index]}}

    def _encode(
        self
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Encode string lines to indices

        Returns:
            source_encoded: encoded ids of source_lines
            target_encoded: encoded ids of target_lines. Unlike source_encoded, <bos> and <eos> are added.
        """
        source_encoded = self.tokenizer.encode_batch(self.source_lines)
        target_encoded = self.tokenizer.encode_batch(self.target_lines)
        source_ids = [torch.tensor(source.ids) for source in source_encoded]
        target_ids = [torch.tensor(target.ids) for target in target_encoded]
        bos = torch.tensor([self.tokenizer.token_to_id("<bos>")])
        eos = torch.tensor([self.tokenizer.token_to_id("<eos>")])
        for i, target_id in enumerate(target_ids):
            target_id = torch.cat([bos, target_id, eos], dim=0)
            target_ids[i] = target_id
        return source_ids, target_ids

    def preprocess(self, filter_fn: callable = filter_by_length, pad_and_mask_fn: callable = pad_and_mask_sequence) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter, encode, pad, and mask the raw source and target text

        Args:
            source_line: raw text in source language
            target_line: raw text in target language

        Returns:
            source_padded_tokens: tensor of padded tokens of source sentences
            source_masks: masked tensor of source sentences
            source_lengths: token length of source sentences before they are padded
            target_padded_tokens: tensor of padded tokens of target sentences
            target_masks: masked tensor of target sentences
            target_lengths: token length of target sentences before they are padded
        """
        source_tokens, target_tokens = self._encode()
        source_tokens, target_tokens = filter_fn(source_tokens, target_tokens, max_length=self.max_length)
        source_lengths = torch.Tensor([len(t) for t in source_tokens])
        target_lengths = torch.Tensor([len(t) for t in target_tokens])
        source_padded_tokens, source_masks, target_padded_tokens, target_masks = pad_and_mask_fn(source_tokens, target_tokens, padding_value=self.tokenizer.token_to_id("<pad>"))
        return source_padded_tokens, source_masks, source_lengths, target_padded_tokens, target_masks, target_lengths
