from typing import List

import numpy as np
import torch
from evaluate import load
from sacrebleu import sentence_bleu


def compute_bleu(refs: List[str], cands: List[str]):
    # bleu = load("bleu")
    bleu = load("/home/data_91_d/zhuangzy/evaluate/metrics/bleu/bleu.py")
    results = bleu.compute(predictions=cands, references=refs)
    return results['bleu']


def compute_sacrebleu(refs: List[str], cands: List[str]):
    sacrebleu = load("sacrebleu")
    results = sacrebleu.compute(predictions=cands, references=[[ref] for ref in refs])
    return results["score"]


def compute_sentence_bleu(refs: List[str], cands: List[str]):
    score = []
    for ref, cand in zip(refs, cands):
        score.append(sentence_bleu(cand, [ref]).score)
    return np.mean(score)


def compute_meteor(refs: List[str], cands: List[str]):
    meteor = load("meteor")
    results = meteor.compute(predictions=cands, references=refs)
    return results['meteor']


def compute_rouge(refs: List[str], cands: List[str]):
    rouge = load("/home/data_91_d/zhuangzy/evaluate/metrics/rouge/rouge.py")
    results = rouge.compute(predictions=cands, references=refs)
    return results['rougeL']


def compute_ppl(sentences: List[str], model_id="huggingface/gpt2"):
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(
        predictions=sentences, model_id=model_id,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=256
    )
    return results['mean_perplexity']


def compute_ibleu(srcs: List[str], refs: List[str], cands: List[str], alpha=0.8):
    src_bleu = compute_bleu(srcs, cands)
    ref_bleu = compute_bleu(refs, cands)
    score = alpha * ref_bleu - (1 - alpha) * src_bleu
    return score
