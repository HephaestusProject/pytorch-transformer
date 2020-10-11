import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.dataloader import WMT14DataLoader
from src.utils import Config
from src.model.transformer import Transformer


def train(langpair: str, model_type: str):
    is_base = model_type == "base"

    config = Config()
    config.add_model(is_base)

    seed_everything(seed=42)

    save_dir = "results/"
    model_name = f"transformer-{langpair}-{model_type}"
    wandb_logger = WandbLogger(model_name, save_dir=save_dir, version=f"version-{datetime.now().strftime('%d-%m-%Y--%H-%M-%S')}")
    checkpoint_path = Path(save_dir) / wandb_logger.version / "checkpoints"
    checkpoint_callback = ModelCheckpoint(checkpoint_path, verbose=True, save_weights_only=True)

    model = Transformer(langpair, is_base)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    dataloader = WMT14DataLoader(langpair, is_base)

    trainer = Trainer(
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        gpus=-1 if torch.cuda.is_available() else None,
        auto_select_gpus=True,
        log_gpu_memory="all",
        check_val_every_n_epoch=1,
        max_steps=config.model.train_hparams.steps,
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--langpair", '-lp', help="Language pair to translate", required=True)
    parser.add_argument("--model", dest="model_type", choices=["base", "big"], required=True, default="base", help="Transformer model type (e.g., base, big)")

    args = parser.parse_args()
    train(args.langpair, args.model_type)
