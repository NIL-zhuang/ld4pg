import argparse
import os
from glob import glob
from typing import List

import pandas as pd

from eval import compute_bart_score, compute_mis_score
from eval import compute_bleu, compute_sentence_bleu, compute_meteor, compute_ppl, compute_ibleu
from eval import compute_div_n, compute_self_bleu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cand_dir", type=str, required=True, help="candidate file directory, including multiple candidates")
    parser.add_argument("--src", type=str, required=True, help="source file, containing src and ref")
    args = parser.parse_args()
    return args


def evaluate(cand: List[str], src: List[str], ref: List[str]):
    print("Evaluating...")
    # evaluate quality
    print(f"BLEU: {compute_bleu(ref, cand)}")
    print(f"sentenceBLEU: {compute_sentence_bleu(ref, cand)}")
    print(f"Meteor: {compute_meteor(ref, cand)}")
    print(f"PPL: {compute_ppl(cand)}")
    print(f"iBLEU: {compute_ibleu(src, ref, cand, alpha=0.8)}")
    print(f"srcBLEU: {compute_bleu(src, cand)}")

    # evaluate diversity
    print(f"Div-4gram: {compute_div_n(cand, 4)}")

    # evaluate semantic
    print(f"BartScore: {compute_bart_score(cand, src)}")
    print(f"MISScore: {compute_mis_score(cand, src)}")


def evaluate_selfBLEU(cands: List[List[str]]):
    print(f"selfBLEU: {compute_self_bleu(cands)}")


def do_evaluate(cands: List[List[str]], src: List[str], ref: List[str]):
    for idx, cand in enumerate(cands):
        print(f"\n{'=' * 10} Example {idx} {'=' * 10}\n")
        evaluate(cand, src, ref)
    if len(cands) > 1:
        evaluate_selfBLEU(cands)


def main():
    args = parse_args()
    cands = []
    for filename in glob(f"{args.cand_dir}/*.txt"):
        print(f"evaluating {filename}...")
        with open(filename, 'r', encoding='utf-8') as f:
            cand = [line.strip() for line in f.readlines()]
            cands.append(cand)
    ref_file = pd.read_csv(args.src)
    src = ref_file['src'].tolist()
    ref = ref_file['tgt'].tolist()
    do_evaluate(cands, src, ref)


if __name__ == "__main__":
    main()
