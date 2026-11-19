import argparse, collections, re
import numpy as np
from lib import auc, rd

ap = argparse.ArgumentParser(description="Bag-of-words ridge trained on the augmentation rows, scored on the holdout (template leakage).")
ap.add_argument("--holdout", default="benchmarks/holdout/holdout.jsonl")
ap.add_argument("--aug", nargs="+", default=["aug/aug_train.jsonl", "aug/devaug.jsonl"])
a = ap.parse_args()
H = rd(a.holdout); yh = np.array([r["label"] for r in H])
A = [r for f in a.aug for r in rd(f)]
ya = np.array([r.get("aug_y", 0) for r in A])
ya = np.where(ya == 2, 2, np.where(ya == 0, 0, -1))
m = ya != -1; A = [r for r, k in zip(A, m) if k]; ya = ya[m]
tok = lambda t: re.findall(r"[a-z_]{3,}", t.lower())
words = collections.Counter(w for r in A[:4000] for w in set(tok(r["text"])))
vocab = {w: i for i, (w, _) in enumerate(words.most_common(4000))}
def X(rows):
    out = np.zeros((len(rows), len(vocab)))
    for k, r in enumerate(rows):
        for w in tok(r["text"]):
            if w in vocab: out[k, vocab[w]] += 1
    return out
Xa, Xh = X(A), X(H); t = np.where(ya == 2, 1.0, -1.0)
for alpha in (1.0, 10.0):
    wv = np.linalg.solve(Xa.T @ Xa + alpha * np.eye(len(vocab)), Xa.T @ t)
    print(f"bow ridge alpha={alpha}: train AUC {auc(Xa @ wv, ya):.4f} holdout AUC {auc(Xh @ wv, yh):.4f}")
