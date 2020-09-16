import torch.nn as nn

from .modules import (
    FeedForwardNetwork,
    LayerNorm,
    MultiHeadAttention,
    PositionalEncoding,
)
from .utils import get_config, get_configs, load_tokenizer


class Model(nn.Module):
    """Transformer Model

    Attributes:
    """

    def __init__(self):
        super().__init__()
        # TODO: embeddings
        # TODO: Encoder
        # TODO: Decoder
        return None

    def forward(self):
        return None


class Embeddings(nn.Module):
    """Input embeddings with positional encoding
    """

    def __init__(self, langpair):
        super().__init__()
        # TODO: support transformer-base and transformer-big
        config = get_configs("model", "tokenizer", langpair=langpair)
        self.dim_model = config.model.dim_model
        self.vocab_size = config.tokenizer.vocab_size
        tokenizer = load_tokenizer(config.tokenizer)
        padding_idx = tokenizer.token_to_id("<pad>")
        self.embedding_matrix = nn.Embedding(
            self.vocab_size, self.dim_model, padding_idx=padding_idx
        )
        self.scale = self.dim_model ** 0.5
        self.max_len = config.model.max_len
        self.positional_encoding = PositionalEncoding(self.max_len, self.dim_model)

    def forward(self, x) -> nn.Embedding:  # TODO: type of x
        embeddings = self.embedding_matrix(x)
        embeddings *= self.scale
        embeddings = self.positional_encoding(embeddings)
        return embeddings


class Encoder(nn.Module):
    """Base class for transformer encoders
    """

    def __init__(self):
        super().__init__()
        self.config = get_config("model")
        self.num_layers = self.config.num_encoder_layer
        self.embeddings = Embeddings()
        self.mha = MultiHeadAttention(attention_mask=False)
        self.attention_dropout = nn.Dropout(p=self.config.attention_dropout)
        self.ln = LayerNorm(self.config.train_hparams.eps)
        self.ffn = FeedForwardNetwork()
        self.residual_dropout = nn.Dropout(p=self.config.residual_dropout)

    def layer(self, x, mask):
        x = x + self.mha(x, mask)
        x = self.attention_dropout(x)
        x = self.ln(x)
        x = x + self.residual_dropout(self.ffn(x))
        x = self.ln(x)
        return x

    def forward(self, x, mask):
        x = self.embeddings(x)
        for num_layer in self.num_layers:
            x = self.layer(x)
        return x


class Decoder(nn.Module):
    """Base class for transformer decoders

    Attributes:
        target_embeddings:
    """

    def __init__(self, target_embeddings: nn.Embedding):
        super().__init__()
        self.target_embeddings = target_embeddings

    def masked_multihead_attention(self):
        return None

    def multihead_attention(self):
        return None
