from typing import NamedTuple, Optional, Tuple

from torch import Tensor, nn

from src.model.model_utils import get_clones
from src.model.modules import (
    Embeddings,
    FeedForwardNetwork,
    LayerNorm,
    MultiHeadAttention,
)
from src.utils import Config

EncoderOut = NamedTuple(
    "EncoderOut",
    [
        (
            "encoder_out",
            Tensor,
        ),  # the last encoder layer's output (batch_size, max_len, dim_model)
        (
            "encoder_mask",
            Tensor,
        ),  # boolean Tensor where padding elements are indicated by False (batch_size, max_len)
        ("source_tokens", Optional[Tensor]),
    ],
)


class EncoderLayer(nn.Module):
    """Encoder layer block"""

    def __init__(self, is_base: bool = True):
        super().__init__()
        self.config = Config()
        self.config.add_model(is_base)

        self.mha = MultiHeadAttention(masked_attention=False)
        self.attention_dropout = nn.Dropout(p=self.config.model.model_params.dropout)
        self.ln = LayerNorm(self.config.model.train_hparams.eps)
        self.ffn = FeedForwardNetwork()
        self.residual_dropout = nn.Dropout(p=self.config.model.model_params.dropout)

    def forward(self, source_emb: Tensor, source_mask: Tensor) -> Tuple[Tensor, Tensor]:
        source_emb = source_emb + self.mha(
            query=source_emb, key=source_emb, value=source_emb
        )
        source_emb = self.attention_dropout(source_emb)
        source_emb = self.ln(source_emb)
        source_emb = source_emb + self.residual_dropout(self.ffn(source_emb))
        source_emb = self.ln(source_emb)
        return source_emb, source_mask


class Encoder(nn.Module):
    """Transformer encoder consisting of EncoderLayers

    Attributes:
        langpair: Language pair to translate. Necessary due to the padding idx of tokenizer.
    """

    def __init__(self, langpair: str, is_base: bool = True) -> None:
        super().__init__()
        self.embedding = Embeddings(langpair)
        self.config = Config()
        self.config.add_model(is_base)
        self.num_layers = self.config.model.model_params.num_encoder_layer
        self.encoder_layers = get_clones(EncoderLayer(), self.num_layers)

    def forward(self, source_tokens: Tensor, source_mask: Tensor) -> NamedTuple:
        source_emb = self.embedding(source_tokens)
        for i in range(self.num_layers):
            source_emb, source_mask = self.encoder_layers[i](source_emb, source_mask)
        return EncoderOut(
            encoder_out=source_emb,
            encoder_mask=source_mask,
            source_tokens=source_tokens,
        )
