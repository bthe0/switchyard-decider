import argparse, time
import numpy as np, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from lib import device, rd
from switchyard_decider.serve import encode

# End to end per request as served (head+tail tokenization of up to 20,000 chars, forward, ridge blend), batch 1.
ap = argparse.ArgumentParser(description="Served-path latency on real prompts (default: the frozen holdout).")
ap.add_argument("model_dir"); ap.add_argument("--rows", default="benchmarks/holdout/holdout.jsonl")
a = ap.parse_args()
dev = device()
tok = AutoTokenizer.from_pretrained(a.model_dir)
m = AutoModelForSequenceClassification.from_pretrained(a.model_dir, dtype=torch.float32 if dev == "cpu" else torch.float16).to(dev).eval()
c = dict(np.load(f"{a.model_dir}/calib.npz")); hid = {}
m.model.register_forward_hook(lambda mod, i, o: hid.__setitem__("h", o[0]))


def one(text):
    with torch.no_grad():
        lg = m(input_ids=torch.tensor([encode(tok, text)], device=dev)).logits[0].float().cpu(); h = hid["h"][0].float()
        e = torch.cat([h[0], h.mean(0)]).cpu().numpy()
    pr = (lg[:3].softmax(-1) + lg[3:6].softmax(-1)) / 2
    old = (np.searchsorted(c["aj"], pr[2].item()) + np.searchsorted(c["g"], -lg[6].item())) / 2 / len(c["aj"])
    if "ra" not in c:
        return old
    q = np.searchsorted(c["rs"], e @ c["ra"] + c["rb"]) / len(c["rs"])
    return (old + c["rw"] * q) / (1 + c["rw"])


texts = [r["text"] for r in rd(a.rows)]
for t in texts[:10]: one(t)
ts = []
for t in texts:
    st = time.perf_counter(); one(t); ts.append((time.perf_counter() - st) * 1000)
ts = np.array(ts)
print(f"{dev} ms p50 {np.percentile(ts, 50):.1f} p90 {np.percentile(ts, 90):.1f} max {ts.max():.1f} ({len(ts)} prompts)")
