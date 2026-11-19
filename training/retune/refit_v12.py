import argparse, json
import numpy as np
from lib import auc, rd, refit, served_h

ap = argparse.ArgumentParser(description="Refit calib.npz and the ridge probe on a train4.py dump; prints val2 and dev-aug AUC.")
ap.add_argument("dump", help="<run>_dump.npz from train4.py"); ap.add_argument("v11_calib", help="v1.1 calib.npz (only `out` is reused)")
ap.add_argument("--val2", default="val2.jsonl"); ap.add_argument("--et-meta", default="tune/et_meta.json")
ap.add_argument("--aug-train", default="aug/aug_train.jsonl"); ap.add_argument("--devaug", default="aug/devaug.jsonl")
ap.add_argument("--out-prefix", required=True, help="writes <prefix>_aw<None|1|3>.npz")
a = ap.parse_args()
c0 = dict(np.load(a.v11_calib))
yv = np.array([r["label"] for r in rd(a.val2)])
L = np.array([r["label"] for r in json.load(open(a.et_meta))])
lab = lambda rows: (lambda y: np.where(y == 2, 2, np.where(y == 0, 0, -1)))(np.array([r.get("aug_y", r["gp"] >= .5 and 2 or 0) for r in rows]))
ya, yd = lab(rd(a.aug_train)), lab(rd(a.devaug))
d = np.load(a.dump)
base = {"out": c0["out"].copy()}
for aw in (None, 1, 3):
    ea = None if aw is None else d["augtr_e"][ya != -1]
    c, info = refit(base, d["val2_z"], d["val2_e"], yv, d["et_e"], L, ea, ya[ya != -1] if aw else None, aw or 0)
    hv, hd = served_h(c, d["val2_z"], d["val2_e"]), served_h(c, d["devaug_z"], d["devaug_e"])
    print("aw", aw, "val2", round(auc(hv, yv), 4), "dev-aug", round(auc(hd, yd), 4), info)
    np.savez(f"{a.out_prefix}_aw{aw}.npz", **c)
