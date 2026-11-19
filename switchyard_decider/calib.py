import argparse

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build calib.npz (reference arrays for serve.py) from validation logits.")
    ap.add_argument("model_dir")
    ap.add_argument("val_logits", help="validation logits .npy, shape (n, 7)")
    # Without a target, output is 1 - hardness percentile; with one, outputs follow that distribution
    # (e.g. a previous decider's raw scores, so existing routing cutoffs keep the same share).
    ap.add_argument("target_p", nargs="?", help="optional .npy of target output scores")
    a = ap.parse_args(argv)
    import torch

    z = torch.tensor(np.load(a.val_logits)).float()
    pr = (z[:, :3].softmax(-1) + z[:, 3:6].softmax(-1)) / 2
    aj, g = np.sort(pr[:, 2].numpy()), np.sort(-z[:, 6].numpy())
    h = (np.searchsorted(aj, pr[:, 2].numpy()) + np.searchsorted(g, -z[:, 6].numpy())) / 2 / len(aj)
    out = np.sort(np.load(a.target_p)) if a.target_p else np.linspace(0, 1, len(h))
    np.savez(f"{a.model_dir}/calib.npz", aj=aj, g=g, h=np.sort(h), out=out)
    print("calib", len(h), "out range", out.min(), out.max())


if __name__ == "__main__":
    main()
