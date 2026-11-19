import json, sys, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# bench_speed.py <model_dir> <device: mps|cuda|cpu> [prompts.jsonl]: forward latency at fixed lengths, then on real prompts.
d, dev = sys.argv[1], sys.argv[2]
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForSequenceClassification.from_pretrained(d, dtype=torch.float32 if dev == "cpu" else torch.float16).to(dev).eval()
print(f"load {time.perf_counter() - t0:.1f}s on {dev}")


def run(x):
    t = time.perf_counter()
    with torch.no_grad():
        m(**x).logits[0].float().cpu()
    return (time.perf_counter() - t) * 1000


for n in (32, 128, 256, 512, 768):
    x = {k: v.to(dev) for k, v in tok("hi " * n, truncation=True, max_length=n, return_tensors="pt").items()}
    for _ in range(3):
        run(x)
    ms = [run(x) for _ in range(20)]
    print(f"{n} tokens: p50 {np.median(ms):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms")
if len(sys.argv) > 3:
    texts = [json.loads(l)["text"][:3000] for l in open(sys.argv[3])]
    ms, lens = [], []
    for t in texts:
        s = time.perf_counter()
        x = tok(t, truncation=True, max_length=768, return_tensors="pt")
        lens.append(x["input_ids"].shape[1])
        run({k: v.to(dev) for k, v in x.items()})
        ms.append((time.perf_counter() - s) * 1000)
    print(f"{len(texts)} prompts (median {np.median(lens):.0f} tokens): p50 {np.median(ms):.1f} ms, "
          f"p95 {np.percentile(ms, 95):.1f} ms, p99 {np.percentile(ms, 99):.1f} ms")
