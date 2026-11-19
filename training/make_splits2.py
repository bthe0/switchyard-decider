import json, glob, hashlib, collections, os
import pyarrow.parquet as pq

def clean(p):
    p = p.strip()
    if p.startswith("["):
        try:
            v = json.loads(p)
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                p = "\n".join(v)
        except ValueError:
            pass
    return p.strip()

JM = {"easy": 0, "medium": 1, "hard": 2}
AM = {"easy": 0, "tie": 1, "hard": 2}
ANN = os.environ.get("ANN", "luna,haiku,qwen,nemotron").split(",")
SLUG = {"haiku": "anthropic_claude-haiku-5.5", "qwen": "qwen_qwen3.8-flash",
        "nemotron": "nvidia_nemotron-3-ultra-550b-a55b_free"}
key = lambda t: hashlib.sha1(" ".join(t.lower().split()).encode()).hexdigest()

judge = collections.defaultdict(dict)
for r in map(json.loads, open("decider_train.jsonl")):
    judge[r["id"]]["luna"] = r["luna"]
for a, s in SLUG.items():
    for f in glob.glob(f"or_labels/{s}/*.jsonl"):
        for l in map(json.loads, open(f)):
            if l["label"] in JM:
                judge[l["id"]][a] = l["label"]

def soft(i):
    v = [JM[judge[i][a]] for a in ANN if a in judge.get(i, {})]
    return [sum(x == c for x in v) / len(v) for c in range(3)] if v else [0.0, 0.0, 0.0]

dec = {r["id"]: r for r in map(json.loads, open("decider_train.jsonl"))}
val_keys, out = set(), {"train": [], "val": []}
for r in map(json.loads, open("val.jsonl")):
    val_keys.add(key(r["text"]))
id_of = {key(r["text"]): i for i, r in dec.items()}
for r in map(json.loads, open("val.jsonl")):
    i = id_of[key(r["text"])]
    out["val"].append({**r, "arena_y": AM[r["arena"]], "jsoft": soft(i), "has_j": 1.0})

d = pq.read_table("judge_battles.parquet", columns=["prompt", "winner_model_a", "winner_model_b"]).to_pylist()
seen = set(val_keys)
for i, r in enumerate(d):
    t = dec[i]["text"] if i in dec else clean(r["prompt"])
    k = key(t)
    if k in seen or not 40 <= len(t) <= 6000:
        continue
    seen.add(k)
    a = "hard" if r["winner_model_a"] else "easy" if r["winner_model_b"] else "tie"
    hj = float(i in judge)
    lab = (2 if a == "hard" or dec[i]["luna"] == "hard" else 0 if a == "easy" and dec[i]["luna"] == "easy" else 1) if i in dec else -100
    out["train"].append({"text": t, "label": lab, "weight": 1.0, "arena_y": AM[a], "jsoft": soft(i), "has_j": hj})

for s, rows in out.items():
    with open(f"{s}2.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("text", "label", "weight", "arena_y", "jsoft", "has_j")}, ensure_ascii=False) + "\n")
    print(s, len(rows), "judged", int(sum(r["has_j"] for r in rows)), collections.Counter(r["arena_y"] for r in rows))
