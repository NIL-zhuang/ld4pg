import os

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from ld4pg.data import AbstractDataModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DatasetModule(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            cfg
    ):
        if (("src_max_token_len" in cfg or "tgt_max_token_len" in cfg) and 
            # (cfg.src_max_token_len is None or cfg.tgt_max_token_len is None) and
            cfg.max_token_len is not None
        ):
            source_max_token_len = cfg.max_token_len
            target_max_token_len = cfg.max_token_len
        else:
            source_max_token_len = cfg.src_max_token_len
            target_max_token_len = cfg.tgt_max_token_len
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
        # self.target['input_ids'][self.target['input_ids'] == 0] = -100

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return dict(
            source_text_input_ids=self.source['input_ids'][index],
            source_text_attention_mask=self.source['attention_mask'][index],
            labels=self.target['input_ids'][index],
            labels_attention_mask=self.target['attention_mask'][index]
        )


class DataModule(AbstractDataModule):
    def __init__(
            self,
            cfg,
            tokenizer: PreTrainedTokenizer,
            train_dataset: pd.DataFrame = None,
            valid_dataset: pd.DataFrame = None,
            test_dataset: pd.DataFrame = None,
            inf_train_dataloader: bool = False,
    ):
        super().__init__(cfg, tokenizer)
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.num_workers = self.cfg.num_workers
        self.infinite_dataloader = inf_train_dataloader
        if train_dataset is not None:
            self.train_dataset = DatasetModule(train_dataset, tokenizer, cfg)
        if valid_dataset is not None:
            self.valid_dataset = DatasetModule(valid_dataset, tokenizer, cfg)
        if test_dataset is not None:
            self.test_dataset = DatasetModule(test_dataset, tokenizer, cfg)
