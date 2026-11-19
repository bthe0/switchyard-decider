import json

import numpy as np

from switchyard_decider.serve import encode

rd = lambda f: [json.loads(l) for l in open(f)]


def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def embed(d, texts, mode="ht"):
    # Mirrors serve.py: logits plus CLS+mean embedding of the last hidden state, batch 1. mode=head is the v1.1 input.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = device()
    tok = AutoTokenizer.from_pretrained(d)
    m = AutoModelForSequenceClassification.from_pretrained(d, dtype=torch.float32 if dev == "cpu" else torch.float16).to(dev).eval()
    hid = {}
    m.model.register_forward_hook(lambda mod, i, o: hid.__setitem__("h", o[0]))
    Z, E = [], []
    with torch.no_grad():
        for t in texts:
            ids = tok(t[:3000], truncation=True, max_length=768)["input_ids"] if mode == "head" else encode(tok, t)
            lg = m(input_ids=torch.tensor([ids], device=dev)).logits[0].float().cpu()
            h = hid["h"][0].float()
            Z.append(lg.numpy()); E.append(torch.cat([h[0], h.mean(0)]).cpu().numpy())
    return np.array(Z), np.array(E, dtype=np.float32)


sm = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)


def served_h(c, z, e):
    pr = (sm(z[:, :3]) + sm(z[:, 3:6])) / 2
    old = (np.searchsorted(c["aj"], pr[:, 2]) + np.searchsorted(c["g"], -z[:, 6])) / 2 / len(c["aj"])
    if "ra" not in c:
        return old
    q = np.searchsorted(c["rs"], e @ c["ra"] + c["rb"]) / len(c["rs"])
    return (old + c["rw"] * q) / (1 + c["rw"])


def auc(s, y):
    a, b = s[y == 2], s[y == 0]
    return float((a[:, None] > b[None]).mean() + .5 * (a[:, None] == b[None]).mean())


def refit(c0, zv, ev, yv, et, L_et, ea=None, ya=None, aw=1.0, alphas=(3e4, 1e5, 3e5, 1e6), recal=True):
    # Same procedure as tune/fit_head.py; optional extra ridge rows (aug) with weight aw. Returns calib dict.
    c = dict(c0)
    pr = (sm(zv[:, :3]) + sm(zv[:, 3:6])) / 2
    if recal:
        c["aj"], c["g"] = np.sort(pr[:, 2]), np.sort(-zv[:, 6])
    old = (np.searchsorted(c["aj"], pr[:, 2]) + np.searchsorted(c["g"], -zv[:, 6])) / 2 / len(c["aj"])
    m3 = (L_et == 0) | (L_et == 2)
    Xr, t, wt = et[m3].astype(np.float64), (L_et[m3] == 2).astype(np.float64), np.ones(m3.sum())
    if ea is not None:
        Xr, t, wt = np.vstack([Xr, ea.astype(np.float64)]), np.r_[t, (ya == 2).astype(np.float64)], np.r_[wt, np.full(len(ea), aw)]
    mu = (Xr * wt[:, None]).sum(0) / wt.sum(); sd = np.sqrt(((Xr - mu) ** 2 * wt[:, None]).sum(0) / wt.sum()) + 1e-6
    X = (Xr - mu) / sd; tm = (t * wt).sum() / wt.sum()
    Xw = X * wt[:, None]; G = X.T @ Xw; r = Xw.T @ (t - tm)
    fold = np.random.default_rng(0).integers(0, 5, len(yv)); best = None
    aucm = lambda h, m: auc(h[m], yv[m])
    for alpha in alphas:
        a = np.linalg.solve(G + alpha * np.eye(G.shape[0]), r) / sd; b = -mu @ a
        s = ev.astype(np.float64) @ a + b; sv = np.sort(s); q = np.searchsorted(sv, s) / len(sv)
        cv = np.mean([aucm(old + max((0, .5, 1, 2, 4, 8), key=lambda w: aucm(old + w * q, fold != k)) * q, fold == k) for k in range(5)])
        if best is None or cv > best[0]: best = (cv, a, b, sv, q, alpha)
    cv, a, b, sv, q, alpha = best
    w = max((0, .5, 1, 2, 4, 8), key=lambda w: auc(old + w * q, yv))
    h = (old + w * q) / (1 + w)
    c.update(h=np.sort(h), ra=a.astype(np.float32), rb=np.float32(b), rs=sv.astype(np.float32), rw=np.float32(w))
    return c, dict(alpha=alpha, w=w, cv=cv, val_auc=auc(h, yv))
