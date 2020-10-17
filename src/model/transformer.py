from torch import Tensor, nn

from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.utils import Config


class Transformer(nn.Module):
    """Transformer Model"""

    def __init__(self, langpair: str, is_base: bool = True) -> None:
        super().__init__()
        configs = Config()
        configs.add_tokenizer(langpair)
        configs.add_model(is_base)
        dim_model: int = configs.model.model_params.dim_model
        vocab_size = configs.tokenizer.vocab_size

        self.encoder = Encoder(langpair)
        self.decoder = Decoder(langpair)
        self.linear = nn.Linear(dim_model, vocab_size)

    def forward(
        self,
        source_tokens: Tensor,
        source_mask: Tensor,
        target_tokens: Tensor,
        target_mask: Tensor,
    ):
        encoder_out, encoder_mask, _ = self.encoder(source_tokens, source_mask)
        target_emb, target_mask = self.decoder(
            target_tokens, target_mask, encoder_out, encoder_mask
        )
        output = self.linear(target_emb)
        return output
