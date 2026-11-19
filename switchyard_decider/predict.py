import argparse
import json

import numpy as np

from switchyard_decider.serve import encode


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write raw logits for every row of val.jsonl (CUDA, bf16).")
    ap.add_argument("model_dir")
    ap.add_argument("out", help="output .npy path")
    a = ap.parse_args(argv)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    m = AutoModelForSequenceClassification.from_pretrained(a.model_dir, torch_dtype=torch.bfloat16).cuda().eval()
    rows = [json.loads(line)["text"] for line in open("val.jsonl")]
    res = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            b = tok.pad({"input_ids": [encode(tok, t) for t in rows[i:i + 32]]}, return_tensors="pt").to("cuda")
            res.append(m(**b).logits.float().cpu().numpy())
    np.save(a.out, np.concatenate(res))


if __name__ == "__main__":
    main()
