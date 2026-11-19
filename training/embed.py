import json, sys, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from switchyard_decider.serve import encode
# embed.py <model_dir>: CLS+mean last-layer embeddings for val.jsonl and a 30k train3.jsonl sample, input to fit_head.py.
d = sys.argv[1]
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForSequenceClassification.from_pretrained(d, dtype=torch.float16).to("mps").eval()
def emb(texts):
    out = []
    with torch.no_grad():
        for t in texts:
            h = m.model(input_ids=torch.tensor([encode(tok, t)], device="mps")).last_hidden_state[0].float()
            out.append(torch.cat([h[0], h.mean(0)]).cpu().numpy())
    return np.array(out, dtype=np.float32)
va = [json.loads(l) for l in open("val.jsonl")]
vt = {r["text"] for r in va}
tr = [r for r in map(json.loads, open("train3.jsonl")) if r["text"] not in vt and (r.get("label", -100) != -100 or r.get("has_g"))]
rng = np.random.default_rng(0); tr = [tr[i] for i in rng.choice(len(tr), min(30000, len(tr)), replace=False)]
np.save("tune/ev.npy", emb([r["text"] for r in va])); print("val done", flush=True)
np.save("tune/et.npy", emb([r["text"] for r in tr]))
json.dump([{"label": r.get("label", -100), "has_g": r.get("has_g", 0), "gp": r.get("gp", 0), "arena_y": r.get("arena_y", -100)} for r in tr], open("tune/et_meta.json", "w"))
print("train done", len(tr), flush=True)
