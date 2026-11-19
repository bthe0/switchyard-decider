import json, os, sys, threading, concurrent.futures as cf
src, m = sys.argv[1], sys.argv[2]
sys.argv = sys.argv[:1]
from bench_dec import call, usage, CAP

rows = [json.loads(l) for l in open(src)]
os.makedirs("gold_labels", exist_ok=True)
f = f"gold_labels/{m.replace('/', '_').replace('~', '')}.jsonl"
done = {json.loads(l)["id"] for l in open(f)} if os.path.exists(f) else set()
todo = [r for r in rows if r["id"] not in done]
print(m, "todo", len(todo), flush=True)
out, lock, n, stop = open(f, "a"), threading.Lock(), [0], threading.Event()

def work(r):
    if stop.is_set():
        return
    p, _ = call(m, r["text"])
    with lock:
        if isinstance(p, float):
            out.write(json.dumps({"id": r["id"], "p": p}) + "\n"); out.flush()
        n[0] += 1
        check = n[0] % 1000 == 0
    if check:
        u = usage()
        print(m, n[0], "usage", u, flush=True)
        if u >= CAP:
            stop.set()

with cf.ThreadPoolExecutor(24) as ex:
    list(ex.map(work, todo))
print(m, "CAP" if stop.is_set() else "done", sum(1 for _ in open(f)), "usage", usage(), flush=True)
