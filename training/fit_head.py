import json, sys, numpy as np

# fit_head.py <calib.npz> <out.npz>: ridge probe on embed.py outputs, blended with the head score; writes ra/rb/rs/rw and a new h.
va = [json.loads(l) for l in open("val.jsonl")]; y = np.array([r["label"] for r in va])
sm = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
def auc(h, m=None):
    m = np.ones(len(y), bool) if m is None else m; a, b = h[m & (y == 2)], h[m & (y == 0)]
    return float((a[:, None] > b[None]).mean() + .5 * (a[:, None] == b[None]).mean())
c = dict(np.load(sys.argv[1]))
z = np.load("tune/z_3000_768.npy"); pr = (sm(z[:, :3]) + sm(z[:, 3:6])) / 2
old = (np.searchsorted(c["aj"], pr[:, 2]) + np.searchsorted(c["g"], -z[:, 6])) / 2 / len(c["aj"])
ev, et = np.load("tune/ev.npy").astype(np.float64), np.load("tune/et.npy").astype(np.float64)
L = np.array([r["label"] for r in json.load(open("tune/et_meta.json"))]); m3 = (L == 0) | (L == 2)
mu, sd = et.mean(0), et.std(0) + 1e-6; X = (et[m3] - mu) / sd; t = (L[m3] == 2).astype(np.float64)
fold = np.random.default_rng(0).integers(0, 5, len(y)); best = None
for alpha in (3e4, 1e5, 3e5, 1e6):
    a = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ (t - t.mean())) / sd; b = -mu @ a
    s = ev @ a + b; sv = np.sort(s); q = np.searchsorted(sv, s) / len(sv)
    cv = np.mean([auc(old + max((0, .5, 1, 2, 4, 8), key=lambda w: auc(old + w * q, fold != k)) * q, fold == k) for k in range(5)])
    print(f"alpha {alpha:g}: ridge auc {auc(s):.4f} blend cv {cv:.4f} (old {auc(old):.4f})")
    if best is None or cv > best[0]: best = (cv, a, b, sv, q)
cv, a, b, sv, q = best; w = max((0, .5, 1, 2, 4, 8), key=lambda w: auc(old + w * q))
h = (old + w * q) / (1 + w)
print("blend w", w, "full-val auc", round(auc(h), 4))
c.update(h=np.sort(h), ra=a.astype(np.float32), rb=np.float32(b), rs=sv.astype(np.float32), rw=np.float32(w))
np.savez(sys.argv[2], **c); np.save(sys.argv[2] + ".valh.npy", h)
