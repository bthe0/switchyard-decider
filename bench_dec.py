import json, re, os, sys, random, time, urllib.request, concurrent.futures as cf
import numpy as np

H = {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"}
CAP, N = float(os.environ.get("CAP", "inf")), int(os.environ.get("N", 300))
Q = {"solve": {"type": "noul", "instructions": (
    "Can a solid mid-tier coding assistant complete this request fully and correctly on its own, without needing a "
    "frontier-level model? Routine edits, lookups, explanations, config and simple scripts are yes; deep multi-file "
    "debugging, subtle concurrency, architecture design, novel algorithms or high-stakes migrations are no.")}}
get = lambda u: json.load(urllib.request.urlopen(urllib.request.Request(u, headers=H), timeout=60))
usage = lambda: get("https://openrouter.ai/api/v1/key")["data"]["usage"]

def call(m, t):
    body = {"model": m, "state": {"opening_task": t[:int(os.environ.get("TRUNC", 12000))], "latest_follow_up": ""}, "questions": Q}
    for k in range(4):
        if k:
            time.sleep(2 * k)
        try:
            r = json.load(urllib.request.urlopen(urllib.request.Request(
                "https://openrouter.ai/api/alpha/decisions", json.dumps(body).encode(), H), timeout=60))
            return r["answers"]["solve"]["noul"], r["usage"].get("cost", 0)
        except Exception as e:
            err = getattr(e, "code", str(e)[:60])
            if err in (400, 402, 403, 404):
                break
    return None, err

def auc(s, y):
    a, b = s[y == 0], s[y == 2]
    return float((a[:, None] > b[None]).mean() + .5 * (a[:, None] == b[None]).mean())

def report(m, res):
    ok = [(i, p) for i, p in res.items() if isinstance(p, float)]
    if len(ok) < 30:
        return print(m, "FAILED", len(ok), list(res.values())[:2], flush=True)
    s, y = np.array([p for _, p in ok]), np.array([va[i]["label"] for i, _ in ok])
    code = np.array([va[i]["domain"] == "coding" for i, _ in ok])
    print(m, json.dumps({"n": len(ok), "auc": round(auc(s, y), 4), "auc_code": round(auc(s[code], y[code]), 4)}), flush=True)

if __name__ == "__main__":
    os.makedirs("dec_bench", exist_ok=True)

    va = [json.loads(l) for l in open("val.jsonl")]
    idx = [i for i, r in enumerate(va) if r["label"] in (0, 2)]
    random.Random(0).shuffle(idx)
    idx = sorted(idx[:N])
    models = sys.argv[1:] or [m["id"] for m in get("https://openrouter.ai/api/v1/models?output_modalities=all")["data"]
                              if m["architecture"].get("output_modalities") == ["decisions"]]
    for m in models:
        f = f"dec_bench/{m.replace('/', '_').replace('~', '')}.json"
        if os.path.exists(f):
            report(m, {int(k): v for k, v in json.load(open(f)).items()}); continue
        if usage() >= CAP:
            print("CAP reached"); break
        with cf.ThreadPoolExecutor(12) as ex:
            out = dict(zip(idx, ex.map(lambda i: call(m, va[i]["text"]), idx)))
        res = {i: o[0] if o[0] is not None else o[1] for i, o in out.items()}
        json.dump(res, open(f, "w"))
        report(m, res)

    ok = [i for i in idx]
    for name in ["base_e2.npy", "cur_decider_val.npy"]:
        z = np.load(f"preds/{name}" if os.path.exists(f"preds/{name}") else name)
        if z.ndim == 2:
            e = np.exp(z - z.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True); z = p[:, 0] + .5 * p[:, 1]
        print("REF", name, round(auc(z[idx], np.array([va[i]["label"] for i in idx])), 4))
    print("usage", usage())
