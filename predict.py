import sys, json, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
d, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(d)
m = AutoModelForSequenceClassification.from_pretrained(d, torch_dtype=torch.bfloat16).cuda().eval()
rows = [json.loads(l)["text"] for l in open("val.jsonl")]
res = []
with torch.no_grad():
    for i in range(0, len(rows), 32):
        b = tok(rows[i:i + 32], truncation=True, max_length=768, padding=True, return_tensors="pt").to("cuda")
        res.append(m(**b).logits.float().cpu().numpy())
np.save(out, np.concatenate(res))
