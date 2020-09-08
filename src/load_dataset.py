from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import WMT14Dataset
from .utils import read_lines

root_dir = Path(__file__).parent.parent
dataset_config_dir = root_dir / "configs" / "dataset"
model_config_dir = root_dir / "configs" / "model"


class WMT14DataModule(LightningDataModule):
    """Load WMT14 dataset to train and test transformer

    Attributes:
        langpair: language pair to translate
    """

    def __init__(self, langpair: str) -> None:
        super().__init__()
        if langpair in ["de-en", "en-de", "deen", "ende"]:
            self.dataset_config_path = dataset_config_dir / "wmt14.de-en.yaml"
        # TODO: add en-fr
        #  elif langpair in ["en-fr", "fr-en", "enfr", "fren"]:
        #      self.dataset_config_path = dataset_config_dir / "wmt14.en-fr.yaml"
        else:
            raise NotImplementedError(
                f'{langpair} is not supported. Hephaestus project aims to reproduce "Attention is all you need", where transformer is tested on de-en and en-fr.'
            )
        self.langpair = langpair
        self.dataset_config = OmegaConf.load(self.dataset_config_path)
        self.model_config = OmegaConf.load(model_config_dir / "transformers.yaml")

    def setup(self, stage: Optional[str] = None) -> None:
        """Assign dataset for use in dataloaders

        Args:
            stage: decide to load train/val or test
        """
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.source_train = read_lines(self.dataset_config.path.source_train)
            self.source_val = read_lines(self.dataset_config.path.source_dev)
            self.target_train = read_lines(self.dataset_config.path.target_train)
            self.target_val = read_lines(self.dataset_config.path.target_dev)
            assert len(self.source_train) == len(self.target_train)
            assert len(self.source_val) == len(self.target_val)
        # Assign test dataset for use in dataloaders
        if stage == "test" or stage is None:
            self.source_test = read_lines(self.dataset_config.path.source_test)
            self.target_test = read_lines(self.dataset_config.path.target_test)
            assert len(self.source_test) == len(self.target_test)

    def train_dataloader(self) -> DataLoader:
        train_dataset = WMT14Dataset(
            self.langpair, self.source_train, self.target_train
        )
        return DataLoader(
            train_dataset,
            batch_size=self.model_config.train_hparams.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=-1,
        )

    def valid_dataloader(self) -> DataLoader:
        val_dataset = WMT14Dataset(self.langpair, self.source_val, self.target_val)
        return DataLoader(
            val_dataset,
            batch_size=self.model_config.train_hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=-1,
        )

    def test_dataloader(self) -> DataLoader:
        test_dataset = WMT14Dataset(self.langpair, self.source_test, self.target_test)
        return DataLoader(
            test_dataset,
            batch_size=self.model_config.train_hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=-1,
        )
