from typing import List

from eval import compute_bart_score, compute_mis_score
from eval import compute_bleu, compute_meteor, compute_ppl, compute_ibleu
from eval import compute_div_n, compute_self_bleu


def evaluate(cands: List[List[str]], src: List[str], ref: List[str]):
    print("Evaluating...")
    # evaluate quality
    cand = cands[0]
    print(f"BLEU: {compute_bleu(ref, cand)}")
    print(f"Meteor: {compute_meteor(ref, cand)}")
    print(f"PPL: {compute_ppl(cand)}")
    print(f"iBLEU: {compute_ibleu(src, ref, cand, alpha=0.8)}")
    print(f"srcBLEU: {compute_bleu(src, cand)}")

    # evaluate diversity
    print(f"Div-4gram: {compute_div_n(cand, 4)}")
    if len(cands) > 1:
        print(f"selfBLEU: {compute_self_bleu(cands)}")

    # evaluate semantic
    print(f"BartScore: {compute_bart_score(cand, src)}")
    print(f"MISScore: {compute_mis_score(cand, src)}")


def evaluate_single(cands: List[str], src: List[str], ref: List[str]):
    cands = [cands]
    evaluate(cands, src, ref)


def main():
    src = ["I like eating apple", "My name is Huang"]
    ref = ["My favourite food is apple", "Huang is my name"]
    cands = [
        ["I like eating apple", "My name is Huang"],
        ["Apple is good", "My name is Huang"]
    ]
    # evaluate(cands, src, ref)
    evaluate_single(cands[0], src, ref)


if __name__ == "__main__":
    main()
