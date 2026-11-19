import json, os
import numpy as np, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from switchyard_decider.serve import encode

# v1.2 retune: BASE=<v1.1 dir> OUT=r10ht LR=1e-5 TRUNC=ht python train4.py, run in a dir holding train4.jsonl (build_aug.py),
# val2.jsonl, devaug.jsonl, aug_train.jsonl and et.jsonl (the ridge sample rows). Writes OUT-final/ and OUT_dump.npz.

BASE, OUT, MAXLEN = os.environ["BASE"], os.environ["OUT"], 768
LR, EP, TRUNC = float(os.environ["LR"]), 1, os.environ.get("TRUNC", "ht")
JW = GW = 2.0
KEEP = ["text", "label", "weight", "arena_y", "jsoft", "has_j", "has_g", "gp"]
tok = AutoTokenizer.from_pretrained(BASE)
SMOKE = os.environ.get("SMOKE")
rd = lambda f: [json.loads(l) for l in open(f)][: 64 if SMOKE else None]


def ids_of(t):
    if TRUNC == "head":
        return tok(t[:3000], truncation=True, max_length=MAXLEN)["input_ids"]
    return encode(tok, t)


def ds(rows):
    d = Dataset.from_list([{k: r.get(k, 0) for k in KEEP} for r in rows])
    return d.map(lambda b: {"input_ids": [ids_of(t) for t in b["text"]], "attention_mask": [[1] * len(ids_of(t)) for t in b["text"]]}, batched=True, remove_columns=["text"])


tr = ds(rd("train4.jsonl"))
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=7, dtype=torch.float32)


class WTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        inputs.pop("weight"); inputs.pop("labels")
        ay, js, hj = inputs.pop("arena_y"), inputs.pop("jsoft"), inputs.pop("has_j")
        hg, gp = inputs.pop("has_g"), inputs.pop("gp")
        out = model(**inputs)
        lg = out.logits.float()
        ce = torch.nn.functional.cross_entropy(lg[:, :3], ay, reduction="none", ignore_index=-100)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(lg[:, 6], gp.float(), reduction="none")
        loss = (ce - JW * hj * (js * lg[:, 3:6].log_softmax(-1)).sum(-1) + GW * hg * bce).mean()
        return (loss, out) if return_outputs else loss


args = TrainingArguments(OUT, num_train_epochs=EP, seed=42, learning_rate=LR, warmup_steps=0.06, weight_decay=0.01,
    per_device_train_batch_size=8, gradient_accumulation_steps=4, bf16=True, train_sampling_strategy="group_by_length",
    max_steps=2 if SMOKE else -1, eval_strategy="no", save_strategy="no", logging_steps=20, report_to=[], dataloader_num_workers=2, remove_unused_columns=False)
t = WTrainer(model=model, args=args, train_dataset=tr.rename_column("label", "labels"), data_collator=DataCollatorWithPadding(tok))
t.train()
model = t.model.half().eval().cuda()
hid = {}
model.model.register_forward_hook(lambda m, i, o: hid.__setitem__("h", o[0]))


def embed(rows):
    ids = [ids_of(r["text"]) for r in rows]
    order = np.argsort([len(i) for i in ids]); Z = np.zeros((len(ids), 7), np.float32); E = np.zeros((len(ids), 2048), np.float16)
    with torch.no_grad():
        for s in range(0, len(ids), 32):
            ix = order[s:s + 32]; L = max(len(ids[i]) for i in ix)
            x = torch.full((len(ix), L), tok.pad_token_id); m = torch.zeros((len(ix), L), dtype=torch.long)
            for k, i in enumerate(ix): x[k, :len(ids[i])] = torch.tensor(ids[i]); m[k, :len(ids[i])] = 1
            x, m = x.cuda(), m.cuda()
            Z[ix] = model(input_ids=x, attention_mask=m).logits.float().cpu().numpy()
            h = hid["h"].float(); mf = m[..., None].float()
            E[ix] = torch.cat([h[:, 0], (h * mf).sum(1) / mf.sum(1)], 1).cpu().numpy()
    return Z, E


res = {}
for n, f in (("val2", "val2.jsonl"), ("devaug", "devaug.jsonl"), ("augtr", "aug_train.jsonl"), ("et", "et.jsonl")):
    z, e = embed(rd(f)); res[n + "_z"], res[n + "_e"] = z, e; print("embedded", n, len(z), flush=True)
np.savez(f"{OUT}_dump.npz", **res)
model.save_pretrained(f"{OUT}-final"); tok.save_pretrained(f"{OUT}-final")
print("TRAINDONE", OUT, flush=True)
