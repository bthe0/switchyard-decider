import json, hashlib, glob
import numpy as np

key = lambda t: hashlib.sha1(" ".join(t.lower().split()).encode()).hexdigest()
gold = {}
for f in glob.glob("gold_labels/*.jsonl"):
    rows = [json.loads(l) for l in open(f)]
    p = np.array([r["p"] for r in rows])
    # Both models label random halves of one pool, so per-model percentiles share a scale.
    pct = (p.argsort().argsort() + .5) / len(p)
    gold.update({r["id"]: float(q) for r, q in zip(rows, pct)})

n = {"train2": 0, "train2_gold": 0, "pool": 0}
with open("train3.jsonl", "w") as out:
    for r in map(json.loads, open("train2.jsonl")):
        g = gold.get("rl:" + key(r["text"])[:16])
        r.update(has_g=float(g is not None), gp=g if g is not None else 0.0)
        out.write(json.dumps(r, ensure_ascii=False) + "\n")
        n["train2"] += 1; n["train2_gold"] += g is not None
    for r in map(json.loads, open("gold_pool.jsonl")):
        if r["id"] in gold:
            out.write(json.dumps({"text": r["text"], "label": -100, "weight": 1.0, "arena_y": -100,
                                  "jsoft": [0.0, 0.0, 0.0], "has_j": 0.0, "has_g": 1.0, "gp": gold[r["id"]]},
                                 ensure_ascii=False) + "\n")
            n["pool"] += 1
print(n, "gold total", len(gold))
