import sys, json, torch, numpy as np, subprocess
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification
va = [json.loads(l) for l in open("val.jsonl")]
y = np.array([r["label"] for r in va])
def score(d):
    subprocess.run(["python", "predict.py", d, "/tmp/s.npy"], check=True)
    z = np.load("/tmp/s.npy"); e = np.exp(z - z.max(1, keepdims=True)); p = e / e.sum(1, keepdims=True); s = p[:, 0] + .5 * p[:, 1]
    a, b = s[y == 0], s[y == 2]
    return float((a[:, None] > b[None]).mean() + .5 * (a[:, None] == b[None]).mean())
def mix(ds, out):
    sd = [load_file(f"{d}/model.safetensors") for d in ds]
    save_file({k: sum(s[k].float() for s in sd).div(len(sd)).to(sd[0][k].dtype) for k in sd[0]}, f"{out}/model.safetensors")
import shutil
runs = sys.argv[1:]
solo = {d: score(d) for d in runs}
print("SOLO", json.dumps(solo), flush=True)
shutil.copytree(runs[0], "soup", dirs_exist_ok=True)
mix(runs, "soup"); print("UNIFORM", score("soup"), flush=True)
order = sorted(runs, key=solo.get, reverse=True)
best, keep = solo[order[0]], [order[0]]
for d in order[1:]:
    mix(keep + [d], "soup"); s = score("soup")
    if s > best: best, keep = s, keep + [d]
mix(keep, "soup"); print("GREEDY", best, keep, flush=True)
