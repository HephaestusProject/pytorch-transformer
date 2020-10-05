import torch.nn as nn

from src.utils import Config


class Model(nn.Module):
    """Transformer Model

    def __init__(self, langpair: str, is_base: bool = True) -> None:
        super().__init__()
        configs = Config()
        configs.add_tokenizer(langpair)
        configs.add_model(is_base)
        dim_model = configs.model.model_params.dim_model
        vocab_size = configs.tokenizer.vocab_size
        )
        self.scale = self.dim_model ** 0.5
        self.max_len = config.model.train_params.max_len
        self.positional_encoding = PositionalEncoding(self.max_len, self.dim_model)

    def forward(self, x) -> nn.Embedding:  # TODO: type of x
        embeddings = self.embedding_matrix(x)
        embeddings *= self.scale
        embeddings = self.positional_encoding(embeddings)
        return embeddings


class Encoder(nn.Module):
    """Base class for transformer encoders"""

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
