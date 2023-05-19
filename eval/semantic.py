from typing import List

import numpy as np
import torch
from evaluate import load
from eval.modules.bart_score import BARTScorer
from eval.modules.mis import MIS


def compute_bart_score(cands: List[str], refs: List[str]):
    bart_scorer = BARTScorer(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint='huggingface/bart-large-cnn'
    )
    bart_scorer.load(path='huggingface/bart-score/bart_score.pth')
    scores = bart_scorer.score(cands, refs, batch_size=128)
    return np.mean(scores)


def compute_mis_score(cand: List[str], refs: List[str]):
    mis = MIS(model_name="huggingface/mis", device='cuda' if torch.cuda.is_available() else 'cpu')
    scores = mis.compute(cand, refs, verbose=True, batch_size=128)
    return np.mean(scores)


def compute_bert_score(sentences: List[str], srcs: List[str]):

    bert_score = load("bert_score")
    return bert_score.compute(cands=sentences, refs=srcs, lang="en", verbose=True, model_type="huggingface/deberta")
