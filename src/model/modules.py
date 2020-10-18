import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from src.utils import Config, load_tokenizer


class PositionalEncoding(nn.Module):
    """PositionalEncoding

    Attributes:
        batch_size: batch size of the input
        max_len: maximum length of the tokens
        embedding_dim: embedding dimension of the given token
    """

    def __init__(self, max_len: int, embedding_dim: int, is_base: bool = True) -> None:
        super().__init__()
        config = Config()
        config.add_model(is_base)

        self.dropout = nn.Dropout(p=config.model.model_params.dropout)
        positional_encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() / embedding_dim * math.log(1e4)
        )
        positional_encoding[:, 0::2] = torch.sin(position / div_term)
        positional_encoding[:, 1::2] = torch.cos(position / div_term)
        positional_encoding = positional_encoding.unsqueeze(
            0
        )  # (1, max_len, embedding_dim)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, embeddings: Tensor) -> Tensor:
        batch_size = embeddings.size(0)
        self.positional_encoding = self.positional_encoding.repeat(batch_size, 1, 1)
        embeddings = (
            embeddings + self.positional_encoding
        )  # (batch_size, max_len, embedding_dim)
        embeddings = self.dropout(embeddings)
        return embeddings


class Embeddings(nn.Module):
    """Input embeddings with positional encoding"""

    def __init__(self, langpair: str, is_base: bool = True) -> None:
        super().__init__()
        # TODO: support transformer-base and transformer-big
        configs = Config()
        configs.add_model(is_base)
        configs.add_tokenizer(langpair)
        tokenizer = load_tokenizer(langpair)
        padding_idx = tokenizer.token_to_id("<pad>")

        self.dim_model: int = configs.model.model_params.dim_model
        self.vocab_size = configs.tokenizer.vocab_size
        self.embedding_matrix = nn.Embedding(
            self.vocab_size, self.dim_model, padding_idx=padding_idx
        )
        self.scale = self.dim_model ** 0.5

    def forward(self, source_tokens: torch.Tensor) -> nn.Embedding:
        """Get embedding matrix for source tokens

        Args:
            source_tokens: (batch_size, max_len)

        Return:
            embeddings: (batch_size, max_len, dim_model)
        """
        embeddings = self.embedding_matrix(
            source_tokens
        )  # (batch_size, max_len, dim_model)
        embeddings *= self.scale
        _, max_len, dim_model = embeddings.size()  # max_len varies with the batch
        positional_encoding = PositionalEncoding(max_len, dim_model)
        embeddings = positional_encoding(embeddings)  # (batch_size, max_len, dim_model)
        return embeddings


class Attention(nn.Module):
    """Compute scaled-dot product attention of Transformer

    Attributes:
        masked_attention: whether to mask attention or not
    """

    def __init__(self, masked_attention: bool = False, is_base: bool = True) -> None:
        super().__init__()
        self.masked_attention = masked_attention
        self.config = Config()
        self.config.add_model(is_base)

        self.dim_q: int = self.config.model.model_params.dim_q
        self.dim_k: int = self.config.model.model_params.dim_k
        self.dim_v: int = self.config.model.model_params.dim_v
        self.dim_model: int = self.config.model.model_params.dim_model
        if self.masked_attention:
            assert (
                self.dim_k == self.dim_v
            ), "masked self-attention requires key, and value to be of the same size"
        else:
            assert (
                self.dim_q == self.dim_k == self.dim_v
            ), "self-attention requires query, key, and value to be of the same size"

        self.q_project = nn.Linear(self.dim_model, self.dim_q)
        self.k_project = nn.Linear(self.dim_model, self.dim_k)
        self.v_project = nn.Linear(self.dim_model, self.dim_v)
        self.scale = self.dim_k ** -0.5

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query: query embedding (batch_size, max_len, dim_model)
            key: key embedding (batch_size, max_len, dim_model)
            value: value embedding (batch_size, max_len, dim_model)
            attention_mask: used to implement masked_attention (batch_size, max_len, max_len)
        """
        q = self.q_project(query)  # (batch_size, max_len, dim_q)
        k = self.k_project(key)  # (batch_size, max_len, dim_k)
        v = self.v_project(value)  # (batch_size, max_len, dim_v)

        qk = (
            torch.bmm(q, k.transpose(1, 2)) * self.scale
        )  # (batch_size, max_len, max_len)
        qk = qk.masked_fill(qk == 0, self.config.model.train_hparams.eps)

        if self.masked_attention:
            qk = qk.masked_fill(
                attention_mask == 0, self.config.model.train_hparams.eps
            )

        attention_weight = torch.softmax(qk, dim=-1)
        attention = torch.matmul(attention_weight, v)  # (batch_size, max_len, dim_v)
        return attention, attention_weight


class MultiHeadAttention(nn.Module):
    """MultiHeadAttention of the Transformer

    Attributes:
        masked_attention: whether to mask attention or not
    """

    def __init__(self, masked_attention: bool = False, is_base: bool = True):
        super().__init__()
        self.attention = Attention(masked_attention)
        config = Config()
        config.add_model(is_base)
        self.batch_size = config.model.train_hparams.batch_size
        self.dim_model: int = config.model.model_params.dim_model
        self.dim_v: int = config.model.model_params.dim_v
        self.num_heads = config.model.model_params.num_heads
        assert (self.dim_model // self.num_heads) == self.dim_v
        assert (
            self.dim_model % self.num_heads == 0
        ), "embed_dim must be divisible by num_heads"
        self.linear = nn.Linear(self.num_heads * self.dim_v, self.dim_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            query: query embedding (batch_size, max_len, dim_model)
            key: key embedding (batch_size, max_len, dim_model)
            value: value embedding (batch_size, max_len, dim_model)
            attention_mask: used to implement masked_attention (batch_size, max_len, max_len)
        """
        heads = [
            self.attention(query, key, value, attention_mask)[0]
            for h in range(self.num_heads)
        ]
        multihead = torch.cat(
            heads, dim=-1
        )  # (batch_size, max_len, dim_model * num_heads)
        multihead = self.linear(multihead)  # (batch_size, max_len, dim_model)
        return multihead


class FeedForwardNetwork(nn.Module):
    def __init__(self, is_base: bool = True):
        super().__init__()
        config = Config()
        config.add_model(is_base)
        self.dim_model: int = config.model.model_params.dim_model
        self.dim_ff: int = config.model.model_params.dim_ff
        self.linear1 = nn.Linear(self.dim_model, self.dim_ff, bias=True)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(self.dim_ff, self.dim_model, bias=True)

    def forward(self, embeddings: Tensor) -> Tensor:
        ffn = self.linear1(embeddings)  # (batch_size, max_len, dim_ff)
        ffn = self.ReLU(ffn)
        ffn = self.linear2(ffn)  # (batch_size, max_len, dim_model)
        return ffn


class LayerNorm(nn.Module):
    def __init__(self, is_base: bool = True, eps: float = 1e-6):
        super().__init__()
        config = Config()
        config.add_model(is_base)
        self.dim_model: int = config.model.model_params.dim_model
        self.gamma = nn.Parameter(torch.ones(self.dim_model))
        self.beta = nn.Parameter(torch.zeros(self.dim_model))
        self.eps = eps

    def forward(self, embeddings: Tensor) -> Tensor:
        mean = torch.mean(embeddings, dim=-1, keepdim=True)  # (batch_size, max_len, 1)
        std = torch.std(embeddings, dim=-1, keepdim=True)  # (batch_size, max_len, 1)
        ln = (
            self.gamma * (embeddings - mean) / (std + self.eps) + self.beta
        )  # (batch_size, max_len, dim_model)
        return ln
