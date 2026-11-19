import sys, json, numpy as np
va = [json.loads(l) for l in open("val.jsonl")]
y = np.array([r["label"] for r in va]); ar = np.array([r["arena"] for r in va])
code = np.array([r["domain"] == "coding" for r in va])
sm = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
def auc(s, pos, neg):  # P(score of easy-side > score of hard-side)
    a, b = s[neg], s[pos]
    return float((a[:, None] > b[None]).mean() + .5 * (a[:, None] == b[None]).mean())
def ps(z, T=1.0):
    if z.ndim == 1: return z
    p = sm(z / T); return p[:, 0] + .5 * p[:, 1]
def fitT(z):  # temperature minimizing NLL of the 3-class labels
    best = min(np.arange(.5, 4.01, .05), key=lambda T: -np.log(sm(z / T)[np.arange(len(y)), y] + 1e-9).mean())
    return float(best)
def report(name, z):
    T = fitT(z) if z.ndim == 2 else 1.0
    s = ps(z, T)
    r = {"auc_lab": auc(s, y == 2, y == 0), "auc_arena": auc(s, ar == "hard", ar == "easy"),
         "auc_code": auc(s, (y == 2) & code, (y == 0) & code), "T": T}
    for share in (.3, .4, .5):  # route the lowest-p_solve share to Claude
        t = np.quantile(s, share); c = s <= t
        r[f"@{int(share*100)}%claude"] = {"hard_to_claude": round(float(c[y == 2].mean()), 3),
                                          "easy_to_go": round(float((~c)[y == 0].mean()), 3),
                                          "arenahard_to_claude": round(float(c[ar == "hard"].mean()), 3)}
    c = s < .5
    r["shim@0.5"] = {"claude_share": round(float(c.mean()), 3), "hard_to_claude": round(float(c[y == 2].mean()), 3),
                     "easy_to_go": round(float((~c)[y == 0].mean()), 3)}
    print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
zs = {f: np.load(f) for f in sys.argv[1:]}
for f, z in zs.items(): report(f, z)
two = [z for z in zs.values() if z.ndim == 2]
if len(two) > 1:
    report("ENSEMBLE", np.log(np.mean([sm(z / fitT(z)) for z in two], 0)))
