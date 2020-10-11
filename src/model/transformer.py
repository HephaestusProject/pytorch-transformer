import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn

from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.utils import Config


class Transformer(LightningModule):
    """Transformer Model"""

    def __init__(self, langpair: str, is_base: bool = True) -> None:
        super().__init__()
        self.configs = Config()
        self.configs.add_tokenizer(langpair)
        self.configs.add_model(is_base)
        dim_model: int = self.configs.model.model_params.dim_model
        vocab_size = self.configs.tokenizer.vocab_size

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

    def training_step(self, batch):
        source = batch.source
        target = batch.target
        target_hat = self(source.padded_token, source.mask, target.padded_token, target.mask)
        loss = nn.CrossEntropyLoss(target_hat, target)  # TODO: fix
        return loss

    def test_step(self, batch):
        source = batch.source
        target = batch.target
        target_hat = self(source.padded_token, source.mask, target.padded_token, target.mask)
        loss = nn.CrossEntropyLoss(target_hat, target)  # TODO: fix
        return loss

    def configure_optimizers(self):

        def _inverse_sqrt_warmup_scheduler(dim_model, current_step, warmup_steps):
            if current_step == 0:
                return 0
            else:
                option_1 = current_step ** (-0.5)
                option_2 = current_step * (warmup_steps ** (-1.5))
                return dim_model ** (-0.5) * min(option_1, option_2)

        optimizer = torch.optim.Adam(self.parameters(), betas=(self.configs.model.train_hparams.beta_1, self.configs.model.train_hparams.beta_2), eps=self.configs.model.train_hparams.eps)
        lr_lambda = lambda step: _inverse_sqrt_warmup_scheduler(self.configs.model.model_params.dim_model, step, self.configs.model.train_hparams.warmup_steps)  # noqa: E731
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]
