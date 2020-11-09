import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from src.dataloader import WMT14DataLoader
from src.model.transformer import Transformer
from src.utils import Config

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disabling parallelism to avoid deadlocks


def train(langpair: str, model_type: str):
    is_base = model_type == "base"

    config = Config()
    config.add_model(is_base)

    seed_everything(seed=42)

    save_dir = "results/"
    model_name = f"transformer-{langpair}-{model_type}"
    wandb_logger = WandbLogger(
        model_name,
        save_dir=save_dir,
        version=f"version-{datetime.now().strftime('%Y-%m-%d--%H-%M-%S')}",
    )
    checkpoint_path = Path(save_dir) / model_name / wandb_logger.version
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path / "{epoch:02d}-{step}--{valid_loss:.2f}-{valid_bleu:.2f}" ,verbose=True, save_weights_only=True, monitor='valid_loss'
    )

    model = Transformer(langpair, is_base)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    dataloader = WMT14DataLoader(langpair, is_base)

    wandb_logger.watch(model, log='gradients', log_freq=100)
    trainer = Trainer(
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        gpus=-1 if torch.cuda.is_available() else None,
        log_gpu_memory="all",
        check_val_every_n_epoch=1,
        val_check_interval=0.01,
        max_steps=config.model.train_hparams.steps,
        weights_summary="top",
        gradient_clip_val=0.1
    )
    dataloader.setup("fit")
    trainer.fit(model, train_dataloader=dataloader.train_dataloader(), val_dataloaders=dataloader.val_dataloader())
    print(f"best model path: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--langpair", "-lp", help="Language pair to translate", required=True
    )
    parser.add_argument(
        "--model",
        dest="model_type",
        choices=["base", "big"],
        required=True,
        default="base",
        help="Transformer model type (e.g., base, big)",
    )

    args = parser.parse_args()
    train(args.langpair, args.model_type)
