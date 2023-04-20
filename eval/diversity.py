from typing import List

import nltk
import numpy as np
from evaluate import load

from .quality import compute_bleu
from eval.modules.tokenizer_13a import Tokenizer13a


def compute_wordcount(sentences: List[str]):
    wordcount = load("word_count")
    counts = wordcount.compute(data=sentences)
    return counts['unique_words']


def compute_div_n(sentences: List[str], ngram=4, tokenizer=Tokenizer13a()):
    dist_list = []
    hyp_ngrams = []
    for sent in sentences:
        hyp_ngrams += nltk.ngrams(tokenizer(sent), ngram)
    total_ngrams = len(hyp_ngrams)
    unique_ngrams = len(list(set(hyp_ngrams)))
    if total_ngrams == 0:
        return 0
    dist_list.append(unique_ngrams / total_ngrams)
    return np.mean(dist_list)


def compute_self_bleu(sentences: List[List[str]]):
    """
    sentences: [cands0, cands1, cands2, ...]
    """
    cands, refs = list(), list()
    for idx, sentence in enumerate(sentences):
        cands += sentence
        t_ref = sentences[:idx] + sentences[idx + 1:]
        ref = list(map(list, zip(*t_ref)))
        refs += ref
    return compute_bleu(refs, cands)
