import argparse, collections
import numpy as np
from lib import auc, rd

ap = argparse.ArgumentParser(description="Holdout length shortcut check: rows per cell, length-only AUC, chars per cell.")
ap.add_argument("--holdout", default="benchmarks/holdout/holdout.jsonl")
H = rd(ap.parse_args().holdout)
cell = np.array([r["cell"] for r in H]); y = np.array([r["label"] for r in H]); L = np.array([len(r["text"]) for r in H]); m = y != 1
print("rows", len(H), collections.Counter(cell.tolist()))
print("length-only AUC (hard vs easy):", round(auc(L[m].astype(float), y[m]), 4))
print("chars per cell p10/p50/p90:", {c: [int(np.percentile(L[cell == c], q)) for q in (10, 50, 90)] for c in sorted(set(cell.tolist()))})
