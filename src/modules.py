import math

import torch
from torch import Tensor, nn

from .utils import get_config


class PositionalEncoding(nn.Module):
    """PositionalEncoding

    Attributes:
        max_len: maximum length of the tokens
        embedding_dim: embedding dimension of the given token
    """
    def __init__(self, max_len: int, embedding_dim: int) -> None:
        super().__init__()
        config = get_config('model')
        self.dropout = nn.Dropout(p=config.pe_dropout)
        positional_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() / embedding_dim * math.log(1e4))
        positional_encoding[:, 0::2] = torch.sin(position / div_term)
        positional_encoding[:, 1::2] = torch.cos(position / div_term)
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)  # (max_len, 1, embedding_dim)
        self.register_buffer('positional_encoding', positional_encoding)  # TODO: register_buffer?

    def forward(self, embeddings: Tensor) -> Tensor:
        embeddings = embeddings + self.positional_encoding  # (batch_size, max_len, embedding_dim)
        embeddings = self.dropout(embeddings)
        return embeddings


class Attention(nn.Module):
    """Compute scaled-dot product attention of Transformer

    Attributes:
        attention_mask: whether to mask attention or not
    """
    def __init__(self, attention_mask: bool = False) -> None:
        super().__init__()
        self.attention_mask = attention_mask
        self.config = get_config('model')
        self.dim_q = self.config.dim_q
        self.dim_k = self.config.dim_k
        self.dim_v = self.config.dim_v
        self.dim_model = self.config.dim_model
        if attention_mask:
            assert (self.dim_k == self.dim_v), "masked self-attention requires key, and value to be of the same size"
        else:
            assert (self.dim_q == self.dim_k == self.dim_v), "self-attention requires query, key, and value to be of the same size"

        self.q_project = nn.Linear(self.dim_model, self.dim_q)
        self.k_project = nn.Linear(self.dim_model, self.dim_k)
        self.v_project = nn.Linear(self.dim_model, self.dim_v)
        self.scale = self.dim_k ** -0.5
        self.dropout = nn.Dropout(self.config.attention_dropout)

    def forward(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        # TODO: masked-attention in decoder
        q = self.q_project(embeddings)  # (batch_size, max_len, dim_q)
        k = self.k_project(embeddings)  # (batch_size, max_len, dim_k)
        v = self.v_project(embeddings)  # (batch_size, max_len, dim_v)
        qk = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (batch_size, max_len, max_len)
        qk = qk.masked_fill(mask == 0, self.config.train_hparams.eps)
        attention_weight = torch.softmax(qk, dim=-1)
        attention = torch.matmul(attention_weight, v)  # (batch_size, max_len, dim_v)
        attention = self.dropout(attention)
        return attention, attention_weight


class MultiHeadAttention(nn.Module):
    """MultiHeadAttention of the Transformer

    Attributes:
        attention_mask: whether to mask attention or not
    """
    def __init__(self, attention_mask: bool = False):
        super().__init__()
        self.attention = Attention(attention_mask)
        config = get_config('model')
        self.batch_size = config.train_hparams.batch_size
        self.dim_model = config.dim_model
        self.dim_v = config.dim_v
        self.num_heads = config.num_heads
        assert (self.dim_model // self.num_heads) == self.dim_v
        assert (self.dim_model % self.num_heads == 0), "embed_dim must be divisible by num_heads"
        self.linear = nn.Linear(self.num_heads * self.dim_v, self.dim_model)

    def forward(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        heads = [self.attention(embeddings, mask)[0] for h in range(self.num_heads)]
        multihead = torch.cat(heads, dim=-1)  # (batch_size, max_len, dim_model * num_heads)
        multihead = self.linear(multihead)  # (batch_size, max_len, dim_model)
        return multihead


class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        config = get_config('model')
        self.dim_model = config.dim_model
        self.dim_ff = config.dim_ff
        self.linear1 = nn.Linear(self.dim_model, self.dim_ff, bias=True)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(self.dim_ff, self.dim_model, bias=True)

    def forward(self, embeddings: Tensor) -> Tensor:
        ffn = self.linear1(embeddings)  # (batch_size, max_len, dim_ff)
        ffn = self.ReLU(ffn)
        ffn = self.linear2(ffn)  # (batch_size, max_len, dim_model)
        return ffn


class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        config = get_config('model')
        self.dim_model = config.dim_model
        self.gamma = nn.Parameter(torch.ones(self.dim_model))
        self.beta = nn.Parameter(torch.zeros(self.dim_model))
        self.eps = eps

    def forward(self, embeddings: Tensor) -> Tensor:
        mean = torch.mean(embeddings, dim=-1, keepdim=True)  # (batch_size, max_len, 1)
        std = torch.std(embeddings, dim=-1, keepdim=True)  # (batch_size, max_len, 1)
        ln = self.gamma * (embeddings - mean) / (std + self.eps) + self.beta  # (batch_size, max_len, dim_model)
        return ln
