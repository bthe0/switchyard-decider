import argparse, collections, json, re
import numpy as np
from lib import auc, rd

ap = argparse.ArgumentParser(description="Holdout leakage QA: exact and 8-gram near-duplicates vs training files, TF-IDF CV AUC.")
ap.add_argument("--holdout", default="benchmarks/holdout/holdout.jsonl")
ap.add_argument("against", nargs="+", help="jsonl files to check against (train3, aug_train, devaug, fresh100 ...)")
a = ap.parse_args()
H = rd(a.holdout)
norm = lambda t: re.sub(r"\s+", " ", t.lower()).strip()
def grams(t, n=8):
    w = norm(t).split()
    return [hash(" ".join(w[i:i + n])[:200]) for i in range(len(w) - n + 1)] if len(w) >= n else [hash(norm(t))]
G = [set(grams(r["text"])) for r in H]
hits = np.zeros(len(H), int)
files = a.against
pool = set().union(*G)
seen = collections.defaultdict(int)
exact = set()
for f in files:
    for r in map(json.loads, open(f)):
        t = r["text"]
        if t in exact: continue
        exact.add(t)
        for g in grams(t):
            if g in pool: seen[g] += 1
for i, g in enumerate(G):
    hits[i] = sum(1 for x in g if seen.get(x))
frac = np.array([len(g) and hits[i] / len(g) for i, g in enumerate(G)])
print("exact matches:", sum(1 for r in H if r["text"] in exact))
print("near-dup >=50% 8-gram:", int((frac >= .5).sum()), "max frac", round(frac.max(), 3))
# Report only: the frozen holdout is never rewritten (the one QA pass that built it dropped 0 rows).
print("near-duplicate rows:", [(i, H[i]["cell"]) for i in range(len(H)) if frac[i] >= .5])
# TF-IDF logistic 5-fold CV
y2 = np.array([r["label"] for r in H]); m2 = y2 != 1
texts = [norm(r["text"]) for r in H]
tf = [collections.Counter(re.findall(r"[a-z0-9_]{2,}", t)) for t in texts]
df = collections.Counter(w for c in tf for w in c)
vocab = {w: i for i, (w, _) in enumerate(df.most_common()[:6000])}
idf = np.array([np.log((1 + len(H)) / (1 + df[[w for w, i in vocab.items() if i == j][0]])) for j in range(len(vocab))])
X = np.zeros((len(H), len(vocab)))
for k, c in enumerate(tf):
    for w, v in c.items():
        if w in vocab: X[k, vocab[w]] = (1 + np.log(v)) * idf[vocab[w]]
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
rng = np.random.default_rng(0); fold = rng.integers(0, 5, len(H))
def lr_auc(Xtr, ttr, Xte):
    w = np.zeros(Xtr.shape[1]); b = 0.0
    for _ in range(300):
        p = 1 / (1 + np.exp(-(Xtr @ w + b))); g = Xtr.T @ (p - ttr) / len(ttr) + .01 * w
        w -= g; b -= (p - ttr).mean()
    return 1 / (1 + np.exp(-(Xte @ w + b)))
a = [auc(lr_auc(X[m2 & (fold != k)], y2[m2 & (fold != k)].astype(float), X[m2 & (fold == k)]), y2[m2 & (fold == k)]) for k in range(5)]
print("tfidf logreg 5-fold CV AUC (hard vs easy):", round(float(np.mean(a)), 4), a)
