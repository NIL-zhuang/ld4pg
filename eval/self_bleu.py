import argparse
import os
from glob import glob
from typing import *

import numpy as np
from rich.progress import track
from sacrebleu import sentence_bleu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cand_dir", type=str, required=True)
    args = parser.parse_args()
    return args


def compute_self_bleu(sentence_lists: List[List[str]]):
    """
    sentences: [[cands0, cands1, cands2, ...]]
    """
    min_self_bleu = []
    avg_self_bleu = []
    for sentences in track(sentence_lists):
        cur_bleu = []
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                self_bleu = sentence_bleu(sentences[i], [sentences[j]]).score
                cur_bleu.append(self_bleu)
        min_self_bleu.append(np.min(cur_bleu))
        avg_self_bleu.append(np.mean(cur_bleu))
    return np.mean(min_self_bleu), np.mean(avg_self_bleu)


def main():
    args = parse_args()
    sentence_lists = []
    prefixes = set()
    for filename in glob(f"{args.cand_dir}/*.txt"):
        prefix = filename.split('/')[-1].split('-')[0]
        prefixes.add(prefix)
    print(f"prefixes: {prefixes}")
    for prefix in prefixes:
        for filename in glob(os.path.join(args.cand_dir, f"{prefix}-*.txt")):
            with open(filename, 'r', encoding='utf-8') as f:
                cand = [line.strip() for line in f.readlines()]
                sentence_lists.append(cand)
        print(f"evaluating {prefix}...")
        sentence_lists = list(zip(*sentence_lists))
        min_self_bleu, avg_self_bleu = compute_self_bleu(sentence_lists)
        print(f"Min self bleu of {prefix} is: {min_self_bleu}")
        print(f"Avg self bleu of {prefix} is: {avg_self_bleu}")


if __name__ == '__main__':
    main()
