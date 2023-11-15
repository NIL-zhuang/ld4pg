import os

import math
import omegaconf
import pandas as pd
import torch
from nltk import word_tokenize
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from rich.progress import track

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
            data_path: str = None,
            data: pd.DataFrame = None,
            tokenizer: PreTrainedTokenizer = None,
            cfg: omegaconf.DictConfig = None,
            keyword_mask_ratio: float = 0.15
    ):
        self.source_max_token_len = cfg.max_token_len
        self.target_max_token_len = cfg.max_token_len
        self.keyword_mask_ratio = keyword_mask_ratio
        self.special_token = cfg.special_token

        if data_path is not None and os.path.exists(data_path):
            print(f"loading {data_path} ...")
            data = torch.load(data_path)
            self.source = data['source']
            self.target = data['target']
            self.kw_ids = data['kw_ids']
            self.kw_masks = data['kw_masks']
        else:
            print(f"building {data_path} ...")
            self.build_dataset(tokenizer, data, data_path)

    def build_dataset(self, tokenizer, data, data_path):
        self.source = tokenizer(
            data['src'].tolist(),
            max_length=self.source_max_token_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            add_special_tokens=True
        )
        self.target = tokenizer(
            data['tgt'].tolist(),
            max_length=self.target_max_token_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
            add_special_tokens=True
        )
        # self.target['input_ids'][self.target['input_ids'] == 0] = -100

        self.kw_ids, self.kw_masks = self.build_keyword_mask(
            data['src'].tolist(),
            self.source['input_ids'].clone().detach(),
            self.target['attention_mask'].clone().detach(),
            tokenizer
        )
        data = {
            'source': self.source,
            'target': self.target,
            'kw_ids': self.kw_ids,
            'kw_masks': self.kw_masks
        }

        if data_path is not None:
            torch.save(data, data_path)

    def build_keyword_mask(self, target_sentence, target_kw, target_kw_mask, tokenizer):
        kw_token_ids_list = []
        kw_attention_mask_list = []
        for sent, ids, mask in track(
                zip(target_sentence, target_kw, target_kw_mask),
                total=len(target_sentence), description=f"building keyword dataset"
        ):
            kw_token_ids, kw_attention_mask = self.build_kw_sentence(
                sent, ids, mask, tokenizer, self.keyword_mask_ratio, special_token=self.special_token
            )
            kw_token_ids_list.append(kw_token_ids)
            kw_attention_mask_list.append(kw_attention_mask)
        return torch.stack(kw_token_ids_list), torch.stack(kw_attention_mask_list)

    def __len__(self):
        return len(self.source['input_ids'])

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
    def replace_split_tokens_with_mask(cls, sentence, tokenizer, k: float = 0.15, special_token: str = "pad_token"):
        assert special_token in tokenizer.special_tokens_map.keys()
        kw_tokens = tokenizer.tokenize(sentence)
        # add <BOS> token here
        kw_tokens = [tokenizer.bos_token] + kw_tokens
        words = word_tokenize(sentence)
        num_reserve_words = math.ceil(len(words) * k)
        most_long_token_idx = sorted(range(len(words)), key=lambda i: len(words[i]), reverse=True)[:num_reserve_words]

        token_idx, word_idx = 0, 0
        kw_attention_mask = [0] * len(kw_tokens)
        kw_attention_mask[0] = 1  # <BOS> token

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
                # kw_tokens[token_idx] = tokenizer.special_tokens_map['pad_token']
                kw_tokens[token_idx] = tokenizer.special_tokens_map[special_token]

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
    def build_kw_sentence(
            cls,
            sentence,
            token_id,
            token_mask,
            tokenizer,
            k: float = 0.15,
            special_token: str = "pad_token"
    ):
        """ return masked tokens and attention mask
        replace all masked tokens with <PAD>
        """
        kw_token_ids, kw_attention_masks = cls.replace_split_tokens_with_mask(
            sentence, tokenizer, k, special_token
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
            data_path: str,
            tokenizer: PreTrainedTokenizer,
            train_dataset: pd.DataFrame = None,
            valid_dataset: pd.DataFrame = None,
            test_dataset: pd.DataFrame = None,
            inf_train_dataloader: bool = False,
    ):
        super().__init__(cfg, tokenizer, train_dataset, valid_dataset, test_dataset, inf_train_dataloader)
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.num_workers = self.cfg.num_workers

        kw_ratio = cfg.kw_ratio
        if train_dataset is not None:
            self.train_dataset = ControlNetKeywordDatasetModule(
                os.path.join(data_path, 'train.pth'), train_dataset, tokenizer, cfg, kw_ratio
            )
        if valid_dataset is not None:
            self.valid_dataset = ControlNetKeywordDatasetModule(
                os.path.join(data_path, 'valid.pth'), valid_dataset, tokenizer, cfg, kw_ratio
            )
        if test_dataset is not None:
            self.test_dataset = ControlNetKeywordDatasetModule(
                os.path.join(data_path, 'test.pth'), test_dataset, tokenizer, cfg, kw_ratio
            )
