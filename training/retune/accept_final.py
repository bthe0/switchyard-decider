import argparse
import numpy as np
from lib import auc, rd, served_h

ap = argparse.ArgumentParser(description="Score the chosen run once on the frozen holdout against v1.1 (paired bootstrap, pairs, wrong-side counts).")
ap.add_argument("calib", help="chosen calib.npz from refit_v12.py")
ap.add_argument("holdout_npz", help="embed_set.py output for the holdout with the new model (--mode ht)")
ap.add_argument("v11_holdout_h", help=".npy of served v1.1 hardness on the holdout (same row order)")
ap.add_argument("fresh_npz", help="embed_set.py output for fresh100 with the new model")
ap.add_argument("fresh_rows", help="fresh100 jsonl with labels")
ap.add_argument("--holdout", default="benchmarks/holdout/holdout.jsonl")
ap.add_argument("--save-h", help="write the new served hardness on the holdout to this .npy")
a = ap.parse_args()
c = dict(np.load(a.calib))
H = rd(a.holdout); yh = np.array([r["label"] for r in H]); cell = np.array([r["cell"] for r in H])
d = np.load(a.holdout_npz); h = served_h(c, d["z"], d["e"])
h11 = np.load(a.v11_holdout_h)
yf = np.array([r["label"] for r in rd(a.fresh_rows)])
dfr = np.load(a.fresh_npz)
P = {}
for r, i in zip(H, range(len(H))):
    if r["pair_id"]: P.setdefault(r["pair_id"], {})[r["cell"]] = i
pm = yh != 1
A2, A1 = auc(h[pm], yh[pm]), auc(h11[pm], yh[pm])
rng = np.random.default_rng(0); diffs = []
ym, hm, hm11 = yh[pm], h[pm], h11[pm]
pos, neg = np.where(ym == 2)[0], np.where(ym == 0)[0]
for _ in range(2000):
    p = rng.choice(pos, len(pos), True); n = rng.choice(neg, len(neg), True)
    a2 = (hm[p][:, None] > hm[n][None]).mean() + .5 * (hm[p][:, None] == hm[n][None]).mean()
    a1 = (hm11[p][:, None] > hm11[n][None]).mean() + .5 * (hm11[p][:, None] == hm11[n][None]).mean()
    diffs.append(a2 - a1)
lo = np.percentile(diffs, 2.5)
okp = sum(1 for q in P.values() if h[q["pair_easy"]] < h[q["pair_hard"]])
okp11 = sum(1 for q in P.values() if h11[q["pair_easy"]] < h11[q["pair_hard"]])
print(f"holdout hard-vs-easy v1.2 {A2:.4f} v1.1 {A1:.4f} diff {A2-A1:+.4f} boot95LB {lo:+.4f} {'PASS' if lo > 0 else 'FAIL'}")
print(f"pairs easy<hard v1.2 {okp}/{len(P)} ({okp/len(P):.0%}) v1.1 {okp11}/{len(P)} {'PASS' if okp >= .85*len(P) and okp > okp11 else 'FAIL'}")
worst = {}
for cl in ("short_easy", "short_hard", "long_easy", "long_hard", "pair_easy", "pair_hard"):
    m = cell == cl; bad2 = int((((h < .5) & (yh == 2)) | ((h > .5) & (yh == 0)))[m].sum()); bad1 = int((((h11 < .5) & (yh == 2)) | ((h11 > .5) & (yh == 0)))[m].sum())
    worst[cl] = (bad1, bad2); print(f"  {cl:12s} wrong-side {bad1} -> {bad2}")
ctl = worst["short_easy"][1] - worst["short_easy"][0] <= 2 and worst["long_hard"][1] - worst["long_hard"][0] <= 2
off = worst["short_hard"][1] < worst["short_hard"][0] and worst["long_easy"][1] < worst["long_easy"][0]
print("controls", "PASS" if ctl else "FAIL", "| off-diagonal falls", "PASS" if off else "FAIL")
print("fresh100", round(auc(served_h(c, dfr["z"], dfr["e"]), yf), 4), "ask-last cells v1.2/v1.1 wrong:",
      {cl: (int((((h < .5) & (yh == 2)) | ((h > .5) & (yh == 0)))[cell == cl].sum()), int((((h11 < .5) & (yh == 2)) | ((h11 > .5) & (yh == 0)))[cell == cl].sum())) for cl in ("long_easy", "pair_easy") })
if a.save_h:
    np.save(a.save_h, h)
