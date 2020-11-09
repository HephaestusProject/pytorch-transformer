import torch
from torch import Tensor
from datasets import load_metric

from src.utils import Config, load_tokenizer


class BLEU():
    """Calculate BLEU"""

    def __init__(self, langpair: str):
        self.configs = Config()
        self.tokenizer = load_tokenizer(langpair)
        self.metric = load_metric("sacrebleu")

    def compute(self, target_hat_indices: Tensor, target_indices: Tensor, target_lenghts: Tensor) -> float:
        """
        Args:
            target_hat_indices: model predictions in indices (batch_size, vocab_size, max_len)
            target_indices: reference sentences in indices (batch_size, max_len)
            target_lengths: reference sentences length including bos and eos
        """
        pred_indices = torch.argmax(target_hat_indices, dim=1)
        target_hat_sentences, target_sentences = [], []
        for i in range(target_indices.size(0)):
            real_length = target_lenghts[i] - 2  # remove eos and bos
            real_length = real_length.int()
            pred = pred_indices[i][:real_length]
            target = target_indices[i][:real_length]
            target_hat_sentence = self.tokenizer.decode(pred.cpu().numpy().tolist())
            target_sentence = self.tokenizer.decode(target.cpu().numpy().tolist())
            target_hat_sentences.append(target_hat_sentence)
            target_sentences.append([target_sentence])  # sacrebleu expects reference to be a list of list
        print('')
        print(f'pred: {target_hat_sentence}')
        print(f'ref: {target_sentence}')
        self.metric.add_batch(predictions=target_hat_sentences, references=target_sentences)
        score = self.metric.compute()
        return score['score']

