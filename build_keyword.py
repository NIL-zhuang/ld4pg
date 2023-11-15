from omegaconf import DictConfig, OmegaConf
import os
from typing import *
from rich.progress import track
import argparse
from transformers import AutoTokenizer

from ld4pg.data import get_dataset
from ld4pg.data.controlnet_data_module import ControlNetKeywordDataModule

_WORD_START = 'Ġ'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="conf/controlnet/config_kw_pad_chatgpt.yaml")
    parser.add_argument("--save_path", type=str, default="/home/zhuangzy/")
    args = parser.parse_args()
    return args


def build_dataset(cfg: DictConfig, evaluation=False):
    tokenizer = AutoTokenizer.from_pretrained(cfg.params.tokenizer)
    dataset = get_dataset(cfg.name)
    dataset_module = ControlNetKeywordDataModule(
        cfg=cfg.params,
        tokenizer=tokenizer,
        train_dataset=dataset[0] if not evaluation else None,
        valid_dataset=dataset[1] if not evaluation else None,
        test_dataset=dataset[2] if evaluation else None,
        inf_train_dataloader=False,
    )
    return dataset_module, tokenizer


def get_keywords(dataset, tokenizer):
    tokens = []
    words = []
    for kw_ids, ids in track(zip(dataset.kw_ids, dataset.target['input_ids'])):
        kw_tokens = tokenizer.convert_ids_to_tokens(kw_ids)
        word_list = []
        token_list = []
        try:
            for idx, token in enumerate(kw_tokens):
                if token not in tokenizer.special_tokens_map.values():
                    token = token.rstrip(_WORD_START)
                    if token.startswith(_WORD_START):
                        # a new word
                        word_list.append(token.strip(_WORD_START))
                    elif idx >= 1 and kw_tokens[idx - 1] in tokenizer.special_tokens_map.values():
                        # the first word
                        word_list.append(token)
                    else:
                        # BPE word
                        word_list[-1] = word_list[-1] + token
                    token_list.append(token)
        except Exception as e:
            print(kw_tokens)
            print(tokenizer.convert_ids_to_tokens(ids))
            print(e)
            raise e
        words.append(word_list)
        tokens.append(token_list)
    return tokens, words


def build_file(tokens: List[List[str]], save_path, prefix: str):
    result = ""
    for token_list in tokens:
        result += " ".join(token_list) + "\n"
    with open(os.path.join(save_path, f"{prefix}.txt"), "w") as f:
        f.write(result)


def main(opt: argparse.Namespace):
    cfg: DictConfig = OmegaConf.load(f"{opt.config}")
    dataset, tokenizer = build_dataset(cfg.data, evaluation=True)
    tokens, words = get_keywords(dataset.test_dataset, tokenizer)
    build_file(tokens, opt.save_path, "tokens")
    build_file(words, opt.save_path, "words")


if __name__ == '__main__':
    option = parse_args()
    main(option)
