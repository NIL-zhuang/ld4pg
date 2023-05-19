import os

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from ld4pg.config import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DatasetModule(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            cfg
    ):
        source_max_token_len = cfg.max_token_len
        target_max_token_len = cfg.max_token_len
        self.data = data

        self.source = tokenizer(
            data['src'].tolist(),
            max_length=source_max_token_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            add_special_tokens=True
        )
        self.target = tokenizer(
            data['tgt'].tolist(),
            max_length=target_max_token_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            add_special_tokens=True
        )
        self.target['input_ids'][self.target['input_ids'] == 0] = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return dict(
            source_text_input_ids=self.source['input_ids'][index],
            source_text_attention_mask=self.source['attention_mask'][index],
            labels=self.target['input_ids'][index],
            labels_attention_mask=self.target['attention_mask'][index]
        )


class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            cfg,
            tokenizer: PreTrainedTokenizer,
            train_dataset: pd.DataFrame,
            valid_dataset: pd.DataFrame,
            test_dataset: pd.DataFrame,
            inf_train_dataloader: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.num_workers = self.cfg.num_workers
        self.infinite_dataloader = inf_train_dataloader
        self.train_dataset = DatasetModule(train_dataset, tokenizer, cfg)
        self.valid_dataset = DatasetModule(valid_dataset, tokenizer, cfg)
        self.test_dataset = DatasetModule(test_dataset, tokenizer, cfg)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
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


def infinite_dataloader(data_loader: DataLoader):
    while True:
        for data in data_loader:
            yield data


def get_dataset(dataset: str):
    datasets = [
        os.path.join(DATASET_PATH, dataset, f"{split}.csv")
        for split in ['train', 'valid', 'test']
    ]
    pd_datasets = [pd.read_csv(dataset).astype(str) for dataset in datasets]
    return pd_datasets
