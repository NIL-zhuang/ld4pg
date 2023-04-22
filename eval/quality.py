from typing import List

import torch
from evaluate import load


def compute_bleu(refs: List[str], cands: List[str]):
    bleu = load("bleu")
    results = bleu.compute(predictions=cands, references=refs)
    return results['bleu']


def compute_meteor(refs: List[str], cands: List[str]):
    meteor = load("meteor")
    results = meteor.compute(predictions=cands, references=refs)
    return results['meteor']


def compute_ppl(sentences: List[str], model_id="huggingface/gpt2"):
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=sentences, model_id=model_id, device='cuda' if torch.cuda.is_available() else 'cpu')
    return results['mean_perplexity']


def compute_ibleu(srcs: List[str], refs: List[str], cands: List[str], alpha=0.8):
    src_bleu = compute_bleu(srcs, cands)
    ref_bleu = compute_bleu(refs, cands)
    score = alpha * ref_bleu - (1 - alpha) * src_bleu
    return score
