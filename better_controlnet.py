from sacrebleu import sentence_bleu
from typing import *
import pandas as pd
import argparse
import json


def compute_sentence_bleu(refs: List[str], cands: List[str]):
    results = [
        sentence_bleu(hypo, [ref]).score
        for hypo, ref in zip(cands, refs)
    ]
    return results


def find_best_control(refs: List[str], cands: List[str], prevs: List[str], srcs: List[str]):
    prev_scores = compute_sentence_bleu(refs, prevs)
    curr_scores = compute_sentence_bleu(refs, cands)
    data = list()
    for src, ref, prev, cand, prev_score, curr_score in zip(
            srcs, refs, prevs, cands, prev_scores, curr_scores
    ):
        data.append({
            'src': src,
            'ref': ref,
            'origin': prev,
            'control': cand,
            'score_diff': curr_score - prev_score,
        })

    data = sorted(data, key=lambda x: x['score_diff'], reverse=True)
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--prev", type=str, required=True)
    parser.add_argument("--cand", type=str, required=True)
    args = parser.parse_args()
    return args


def read_file(fpath: str):
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [line.strip('\n') for line in lines]
        return lines


def main(opt: argparse.Namespace):
    src_df = pd.read_csv(opt.src)
    src = src_df['src'].tolist()
    ref = src_df['tgt'].tolist()
    prev = read_file(opt.prev)
    cand = read_file(opt.cand)
    data = find_best_control(ref, cand, prev, src)
    with open('better_controlnet.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    option = parse_args()
    main(option)
