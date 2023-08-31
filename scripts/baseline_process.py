import json
from glob import glob
import pandas as pd
import os


def to_gpvae(df, file):
    df.to_csv(file.replace(".csv", ".tsv"), sep="\t", index=False, header=False)


def to_diffuseq(df, file):
    # to json
    results = []
    for src, tgt in zip(
            df["src"].tolist(),
            df['tgt'].tolist()
    ):
        line = {'src': src, 'trg': tgt}
        results.append(json.dumps(line))
    with open(file.replace(".csv", ".jsonl"), 'w+', encoding='utf-8') as f:
        f.writelines('\n'.join(results))


def to_seqdiffuseq(df, file):
    src_sentences = df['src'].tolist()
    tgt_sentences = df['tgt'].tolist()
    with open(file.replace(".csv", ".src"), 'w+', encoding='utf-8') as f:
        f.writelines('\n'.join(src_sentences))
    with open(file.replace(".csv", ".tgt"), 'w+', encoding='utf-8') as f:
        f.writelines('\n'.join(tgt_sentences))


def main():
    for fname in glob("datasets/twitter/*.csv"):
        print(fname)
        df = pd.read_csv(fname)
        to_diffuseq(df, fname)


if __name__ == '__main__':
    main()
