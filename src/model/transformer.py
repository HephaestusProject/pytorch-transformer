import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.nn import functional as F

from src.model.decoder import Decoder
from src.model.encoder import Encoder
from src.utils import Config, load_tokenizer
from src.optim import WarmUpAdam


class Transformer(LightningModule):
    """Transformer Model"""

    def __init__(self, langpair: str, is_base: bool = True) -> None:
        super().__init__()
        self.configs = Config()
        self.configs.add_tokenizer(langpair)
        self.configs.add_model(is_base)
        self.dim_model: int = self.configs.model.model_params.dim_model
        vocab_size = self.configs.tokenizer.vocab_size
        tokenizer = load_tokenizer(langpair)
        self.padding_idx = tokenizer.token_to_id("<pad>")

        self.encoder = Encoder(langpair)
        self.decoder = Decoder(langpair)
        self.linear = nn.Linear(self.dim_model, vocab_size)

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

    def training_step(self, batch, batch_idx):
        source = batch["source"]
        target = batch["target"]
        target_hat = self(
            source["padded_token"],
            source["mask"],
            target["padded_token"],
            target["mask"],
        )  # (batch_size, max_len, vocab_size)
        target_hat.transpose_(1, 2)  # (batch_size, vocab_size, max_len)
        target["padded_token"] = target["padded_token"][
            :, 1:
        ]  # remove <bos> from target
        target_hat = target_hat[:, :, :-1]  # match shape with target
        loss = F.cross_entropy(
            target_hat, target["padded_token"], ignore_index=self.padding_idx
        )
        self.logger.experiment.log({'train_loss': loss})
        return loss

    def test_step(self, batch, batch_idx):
        source = batch["source"]
        target = batch["target"]
        target_hat = self(
            source["padded_token"],
            source["mask"],
            target["padded_token"],
            target["mask"],
        )  # (batch_size, max_len, vocab_size)
        target_hat.transpose_(1, 2)  # (batch_size, vocab_size, max_len)
        target["padded_token"] = target["padded_token"][
            :, 1:
        ]  # remove <bos> from target
        target_hat = target_hat[:, :, :-1]  # match shape with target
        loss = F.cross_entropy(
            target_hat, target["padded_token"], ignore_index=self.padding_idx
        )
        self.logger.experiment.log({'test_loss': loss})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            betas=(
                self.configs.model.train_hparams.beta_1,
                self.configs.model.train_hparams.beta_2,
            ),
            eps=self.configs.model.train_hparams.eps,
        )
        return optimizer

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # update optimizer every 2 steps if set 12500 tokens per batch (paper: 25000 tokens per step)
        update_period = 25000 // self.configs.model.train_hparams.batch_size
        warmup_steps = self.configs.model.train_hparams.warmup_steps
        global_step = self.trainer.global_step
        if global_step % update_period == 0:
            step_num = global_step // update_period  # lazy update
            lr_scale = np.min([np.power(step_num, -0.5), step_num * np.power(warmup_steps, -1.5)])
            lr = lr_scale * np.power(self.dim_model, -0.5)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad()
