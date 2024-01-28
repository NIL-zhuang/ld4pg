import argparse
import json
import os
from glob import glob
from typing import *

import numpy as np
from nltk import word_tokenize
from tqdm import tqdm
from transformers import AutoTokenizer

KEY_TOKEN_FILE = "tokens.txt"
KEY_WORD_FILE = "words.txt"
TOKENIZER = "huggingface/bart-base"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_file", type=str, default="/home/zhuangzy/controlnet")
    parser.add_argument("--cand_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=False)
    args = parser.parse_args()
    return args


def compute_recall(cands: List[List[str]], oracles: List[List[str]]):
    precision = []
    for cand, oracle in zip(cands, oracles):
        hit = [o for o in oracle if o in cand]
        precision.append(len(hit) / len(oracle))
    return np.round(np.mean(precision), 2)


def evaluate(sentences: List[str], oracle_tokens: List[List[str]], oracle_words: List[List[str]]):
    words = [word_tokenize(sentence) for sentence in sentences]
    word_recall = compute_recall(words, oracle_words)
    tokens = [tokenizer.tokenize(sentence) for sentence in sentences]
    token_recall = compute_recall(tokens, oracle_tokens)
    results = {
        "token_recall": word_recall,
        "word_recall": token_recall
    }
    for k, v in results.items():
        print(f"{k}: {v}")
    return results


def load(filename):
    res = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            res.append(line.strip('\n').split(' '))
    return res


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        res = f.readlines()
        return [line.strip('\n') for line in res]


def main(opt: argparse.Namespace):
    oracle_tokens = load(os.path.join(opt.base_file, KEY_TOKEN_FILE))

    oracle_words = load(os.path.join(opt.base_file, KEY_WORD_FILE))
    eval_results = {}
    for filename in tqdm(sorted(glob(f"{opt.cand_dir}/*.txt")), desc="Evaluating..."):
        print(f"\n\n evaluating {filename}...")
        cand_sentences = read_file(filename)
        result = evaluate(cand_sentences, oracle_tokens, oracle_words)
        eval_results[filename] = result

    result = json.dumps(eval_results, indent=2, ensure_ascii=False)
    print(result)
    if opt.save_path is not None:
        with open(opt.save_path, 'w+', encoding='utf-8') as f:
            f.write(result)


if __name__ == '__main__':
    options = parse_args()
    main(options)
