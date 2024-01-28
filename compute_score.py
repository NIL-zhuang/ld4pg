import argparse
import json
from glob import glob
from typing import List

import pandas as pd
from tqdm import tqdm

from eval import compute_bert_score
from eval import (
    compute_bleu, compute_sentence_bleu, compute_sacrebleu,
    compute_meteor, compute_ppl, compute_ibleu, compute_rouge
)
from eval import compute_div_n


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cand_dir", type=str, required=True,
                        help="candidate file directory, including multiple candidates")
    parser.add_argument("--src", type=str, required=True, help="source file, containing src and ref")
    parser.add_argument("--save_path", type=str, required=False, help="location of evaluation results")
    args = parser.parse_args()
    return args


def evaluate(cand: List[str], src: List[str], ref: List[str]):
    print("Evaluating...")
    results = {
        "BLEU": compute_bleu(ref, cand),
        "sentenceBLEU": compute_sentence_bleu(ref, cand),
        "Rouge": compute_rouge(ref, cand),
        "Meteor": compute_meteor(ref, cand),
        "PPL": compute_ppl(cand),
        "iBLEU": compute_ibleu(src, ref, cand, alpha=0.8),
        "srcBLEU": compute_bleu(src, cand),
        "src-BScore": compute_bert_score(cand, src),
        "tgt-BScore": compute_bert_score(cand, ref),
        "Div-4gram": compute_div_n(cand, 4),
    }
    results['iBScore'] = results['tgt-BScore'] - results['srcBLEU']
    for k, v in results.items():
        print(f"{k}: {v}")
    return results


def do_evaluate(cands: List[str], src: List[str], ref: List[str]):
    print(f"{'=' * 20}")
    result = evaluate(cands, src, ref)
    return result


def main():
    args = parse_args()
    ref_file = pd.read_csv(args.src)
    src = ref_file['src'].tolist()
    ref = ref_file['tgt'].tolist()
    eval_results = {}
    for filename in tqdm(sorted(glob(f"{args.cand_dir}/*.txt"), reverse=True), desc="Evaluating..."):
        print(f"\n\n evaluating {filename}...")
        with open(filename, 'r', encoding='utf-8') as f:
            cand = [line.strip() for line in f.readlines()]
            eval_res = do_evaluate(cand, src, ref)
            eval_results[filename] = eval_res

    result = json.dumps(eval_results, indent=2, ensure_ascii=False)
    print(result)
    if args.save_path is not None:
        with open(args.save_path, 'w+', encoding='utf-8') as f_eval:
            f_eval.write(result)


def test():
    predictions = ["hello there general kenobi", "on our way to ankh morpork"]
    references = ["hello there general kenobi", "goodbye ankh morpork"]
    print(compute_sacrebleu(references, predictions))
    print(compute_bert_score(references, predictions))


if __name__ == "__main__":
    # test()
    main()
