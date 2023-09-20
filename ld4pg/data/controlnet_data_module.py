import math
import os

import pandas as pd
import torch
from nltk import word_tokenize
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from ld4pg.data.data_module import AbstractDataModule

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ControlNetKeywordDatasetModule(Dataset):
    """
    额外提供的信息大约是
    ids:   <BOS> <PAD> <PAD> KEYWORD <PAD> <PAD> KEYWORD <PAD> <EOS> <PAD> ...
    masks:   1    0    0      1      0     0      1      0     1     0   ...
    """
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            cfg,
            keyword_mask_ratio: float = 0.15
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

        self.keyword_mask_ratio = keyword_mask_ratio
        self.kw_ids, self.kw_masks = self.build_keyword_mask(
            data['tgt'].tolist(), self.target['input_ids'], self.target['attention_mask'],
            tokenizer,
        )

    def build_keyword_mask(self, target_sentence, target_kw, target_kw_mask, tokenizer):
        kw_token_ids_list = []
        kw_attention_mask_list = []
        for sent, ids, mask in zip(target_sentence, target_kw, target_kw_mask):
            kw_token_ids, kw_attention_mask = self.build_kw_sentence(
                sent, ids, mask, tokenizer, self.keyword_mask_ratio
            )
            kw_token_ids_list.append(kw_token_ids)
            kw_attention_mask_list.append(kw_attention_mask)
        return kw_token_ids_list, kw_attention_mask_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return dict(
            source_text_input_ids=self.source['input_ids'][index],
            source_text_attention_mask=self.source['attention_mask'][index],
            labels=self.target['input_ids'][index],
            labels_attention_mask=self.target['attention_mask'][index],
            keyword_label_ids=self.kw_ids[index],
            keyword_label_attention_mask=self.kw_masks[index]
        )

    @classmethod
    def replace_split_tokens_with_mask(cls, sentence, tokenizer, k: float = 0.15):
        kw_tokens = tokenizer.tokenize(sentence)
        # add <BOS> token here
        kw_tokens = [tokenizer.bos_token] + kw_tokens
        words = word_tokenize(sentence)
        num_reserve_words = math.ceil(len(words) * k)
        most_long_token_idx = sorted(range(len(words)), key=lambda i: len(words[i]), reverse=True)[:num_reserve_words]

        token_idx, word_idx = 0, 0
        kw_attention_mask = [0] * len(kw_tokens)
        kw_attention_mask[0] = 1    # <BOS> token

        while token_idx < len(kw_tokens) and word_idx < len(words):
            cur_token = kw_tokens[token_idx]
            if cur_token in tokenizer.special_tokens_map.values():
                token_idx += 1
                continue
            cur_token = cur_token.strip('Ġ')
            cur_word = words[word_idx]

            if word_idx in most_long_token_idx:
                kw_attention_mask[token_idx] = 1
            else:
                # 不是关键信息的 token 就替换成 pad
                kw_tokens[token_idx] = tokenizer.special_tokens_map['pad_token']

            if cur_word.startswith(cur_token):
                n_cur_word = cur_word[len(cur_token):]
                if n_cur_word == '':
                    word_idx += 1
                else:
                    words[word_idx] = cur_word[len(cur_token):]
            token_idx += 1

        kw_token_ids = tokenizer.convert_tokens_to_ids(kw_tokens)
        return kw_token_ids, kw_attention_mask

    @classmethod
    def build_kw_sentence(cls, sentence, token_id, token_mask, tokenizer, k: float = 0.15):
        """ return masked tokens and attention mask
        replace all masked tokens with <PAD>
        """
        kw_token_ids, kw_attention_masks = cls.replace_split_tokens_with_mask(
            sentence, tokenizer, k
        )

        kw_token_ids = torch.tensor(kw_token_ids)
        kw_attention_masks = torch.tensor(kw_attention_masks)

        token_id[:len(kw_token_ids)] = kw_token_ids
        token_mask[:len(kw_attention_masks)] = kw_attention_masks
        return token_id, token_mask


class ControlNetKeywordDataModule(AbstractDataModule):
    def __init__(
            self,
            cfg,
            tokenizer: PreTrainedTokenizer,
            train_dataset: pd.DataFrame = None,
            valid_dataset: pd.DataFrame = None,
            test_dataset: pd.DataFrame = None,
            inf_train_dataloader: bool = False
    ):
        super().__init__(cfg, tokenizer, train_dataset, valid_dataset, test_dataset, inf_train_dataloader)
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.num_workers = self.cfg.num_workers
        if train_dataset is not None:
            self.train_dataset = ControlNetKeywordDatasetModule(train_dataset, tokenizer, cfg)
        if valid_dataset is not None:
            self.valid_dataset = ControlNetKeywordDatasetModule(valid_dataset, tokenizer, cfg)
        if test_dataset is not None:
            self.test_dataset = ControlNetKeywordDatasetModule(test_dataset, tokenizer, cfg)
