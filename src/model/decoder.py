from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from src.model.model_utils import get_clones
from src.model.modules import (
    Embeddings,
    FeedForwardNetwork,
    LayerNorm,
    MultiHeadAttention,
)
from src.utils import Config


class DecoderLayer(nn.Module):
    """Decoder layer block"""

    def __init__(self, is_base: bool = True):
        super().__init__()
        self.config = Config()
        self.config.add_model(is_base)

        self.masked_mha = MultiHeadAttention(masked_attention=True)
        self.mha = MultiHeadAttention(masked_attention=False)
        self.ln = LayerNorm(self.config.model.train_hparams.eps)
        self.ffn = FeedForwardNetwork()
        self.residual_dropout = nn.Dropout(p=self.config.model.model_params.dropout)

    def attention_mask(self, batch_size: int, seq_len: int) -> Tensor:
        attention_shape = (batch_size, seq_len, seq_len)
        attention_mask = np.triu(np.ones(attention_shape), k=1).astype("uint8")
        attention_mask = torch.from_numpy(attention_mask) == 0
        return attention_mask  # (batch_size, seq_len, seq_len)

    def forward(
        self,
        target_emb: Tensor,
        target_mask: Tensor,
        encoder_out: Tensor,
        encoder_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            target_emb: input to the decoder layer (batch_size, seq_len, dim_model)
            target_mask: padding mask of the target embedding
            encoder_out: the last encoder layer's output (batch_size, seq_len, dim_model)
            encoder_mask: boolean Tensor where padding elements are indicated by False (batch_size, seq_len)
        """
        attention_mask = self.attention_mask(target_emb.size(0), target_emb.size(1))
        target_emb = target_emb + self.masked_mha(
            query=target_emb,
            key=target_emb,
            value=target_emb,
            attention_mask=attention_mask,
        )
        target_emb = self.ln(target_emb)
        target_emb = target_emb + self.mha(
            query=target_emb, key=encoder_out, value=encoder_out
        )
        target_emb = self.ln(target_emb)
        target_emb = target_emb + self.ffn(target_emb)
        return target_emb, target_mask


class Decoder(nn.Module):
    """Base class for transformer decoders

    Attributes:
        langpair: Language pair to translate. Necessary due to the padding idx of tokenizer.
    """

    def __init__(self, langpair: str, is_base: bool = True) -> None:
        super().__init__()
        self.embedding = Embeddings(langpair)
        self.config = Config()
        self.config.add_model(is_base)
        self.num_layers = self.config.model.model_params.num_decoder_layer
        self.decoder_layers = get_clones(DecoderLayer(), self.num_layers)

    def forward(
        self,
        target_tokens: Tensor,
        target_mask: Tensor,
        encoder_out: Tensor,
        encoder_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            target_emb: input to the decoder layer (batch_size, seq_len, dim_model)
            target_mask: padding mask of the target embedding
            encoder_out: the last encoder layer's output (batch_size, seq_len, dim_model)
            encoder_mask: boolean Tensor where padding elements are indicated by False (batch_size, seq_len)
        """
        target_emb = self.embedding(target_tokens)
        for i in range(self.num_layers):
            target_emb, target_mask = self.decoder_layers[i](
                target_emb, target_mask, encoder_out, encoder_mask
            )
        return target_emb, target_mask
