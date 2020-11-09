import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class LabelSmoothingLoss(_Loss):

    def __init__(self, num_classes: int, epsilon: float) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.confidence = torch.tensor(1 - self.epsilon)
        self.regularization = torch.tensor(self.epsilon / self.num_classes)

    def forward(self, target_hat: Tensor, target: Tensor, ignore_index: int = -100) -> Tensor:
        """
        Args:
            target_hat: model prediction (batch_size, vocab_size, max_len)
            target: vocab index (batch_size, max_len)
            ignore_index: the index does not affect loss
        """
        target_hat = target_hat.log_softmax(dim=1)
        with torch.no_grad():
            true = F.one_hot(target, num_classes=self.num_classes).transpose(1, 2).float()  # (batch_size, vocab_size, max_len)
            true *= self.confidence
            true += self.regularization
            true.masked_fill_(target.unsqueeze(dim=1) == ignore_index, 0)  # to ignore ignore_index prediction
        loss = torch.mean(torch.sum(-true * target_hat, dim=1))
        return loss
