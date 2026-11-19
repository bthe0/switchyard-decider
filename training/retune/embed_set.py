import argparse, time
import numpy as np
from lib import embed, rd

ap = argparse.ArgumentParser(description="Logits and CLS+mean embeddings for every row of a jsonl file, as served.")
ap.add_argument("model_dir"); ap.add_argument("rows", help="jsonl with a text field"); ap.add_argument("out", help=".npz path")
ap.add_argument("--mode", choices=("ht", "head"), default="ht", help="ht = v1.2 head+tail input, head = v1.1 input")
a = ap.parse_args()
t = time.time()
Z, E = embed(a.model_dir, [r["text"] for r in rd(a.rows)], mode=a.mode)
np.savez(a.out, z=Z, e=E); print(a.out, len(Z), round(time.time() - t), "s")
