from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def filter_by_length(source_tokens_list: List[torch.Tensor], target_tokens_list: List[torch.Tensor], max_length: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Filter sentences based on their lengths

    Args:
        source_tokens_list: list of tokens of source sentences
        target_tokens_list: list of tokens of target sentences
        max_length: filter sentences larger than this size

    Returns:
        source_tokens_filtered: list of tokens of source sentences smaller than the max length
        target_tokens_filtered: list of tokens of target sentences smaller than the max length
    """
    source_tokens_filtered, target_tokens_filtered = [], []
    for i in range(len(source_tokens_list)):
        source_tokens = source_tokens_list[i]
        target_tokens = target_tokens_list[i]
        if len(source_tokens) > max_length or len(target_tokens) > max_length:
            continue
        source_tokens_filtered.append(source_tokens)
        target_tokens_filtered.append(target_tokens)
    return source_tokens_filtered, target_tokens_filtered


def pad_and_mask_sequence(source_tokens: List[torch.Tensor], target_tokens: List[torch.Tensor], padding_value: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad and mask sequences

)      Args:
        source_tokens: list of tokens of source sentences
        target_tokens: list of tokens of target sentences
        padding_value: pad token id

    Returns:
        source_padded_tokens: tensor of padded tokens of source sentences
        source_masks: masked tensor of source sentences
        target_padded_tokens: tensor of padded tokens of target sentences
        target_masks: masked tensor of target sentences
    """
    source_padded_tokens = pad_sequence(source_tokens, batch_first=True, padding_value=padding_value)
    target_padded_tokens = pad_sequence(target_tokens, batch_first=True, padding_value=padding_value)
    source_masks = source_padded_tokens.ne(padding_value)
    target_masks = target_padded_tokens.ne(padding_value)
    return source_padded_tokens, source_masks, target_padded_tokens, target_masks
