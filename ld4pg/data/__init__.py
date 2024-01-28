import os

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from ld4pg.config import *


def infinite_dataloader(data_loader: DataLoader):
    while True:
        for data in data_loader:
            yield data


class AbstractDataModule(pl.LightningDataModule):
    def __init__(
            self,
            cfg,
            tokenizer: PreTrainedTokenizer = None,
            train_dataset: pd.DataFrame = None,
            valid_dataset: pd.DataFrame = None,
            test_dataset: pd.DataFrame = None,
            inf_train_dataloader: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = 16
        self.num_workers = 4
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.infinite_dataloader = inf_train_dataloader

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False
        )
        return infinite_dataloader(dataloader) if self.infinite_dataloader else dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False
        )


def get_dataset(dataset: str):
    datasets = [
        os.path.join(DATASET_PATH, dataset, f"{split}.csv")
        for split in ['train', 'valid', 'test']
    ]
    pd_datasets = [pd.read_csv(dataset).astype(str) for dataset in datasets]
    return pd_datasets
