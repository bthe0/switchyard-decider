import sys, numpy as np, torch

# calib.py <model_dir> <val_logits.npy> [target_p.npy]: reference arrays for serve.py.
# Without a target, output is 1 - hardness percentile; with one, outputs follow that distribution
# (e.g. a previous decider's raw scores, so existing routing cutoffs keep the same share).
d, z = sys.argv[1], torch.tensor(np.load(sys.argv[2])).float()
pr = (z[:, :3].softmax(-1) + z[:, 3:6].softmax(-1)) / 2
aj, g = np.sort(pr[:, 2].numpy()), np.sort(-z[:, 6].numpy())
h = (np.searchsorted(aj, pr[:, 2].numpy()) + np.searchsorted(g, -z[:, 6].numpy())) / 2 / len(aj)
out = np.sort(np.load(sys.argv[3])) if len(sys.argv) > 3 else np.linspace(0, 1, len(h))
np.savez(f"{d}/calib.npz", aj=aj, g=g, h=np.sort(h), out=out)
print("calib", len(h), "out range", out.min(), out.max())
