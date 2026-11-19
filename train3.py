import json, os, glob
import numpy as np, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer,
                          TrainingArguments, DataCollatorWithPadding)

BASE, MAXLEN = os.environ.get("BASE", "answerdotai/ModernBERT-large"), int(os.environ.get("MAXLEN", 768))
LS, SEED = float(os.environ.get("LS", 0)), int(os.environ.get("SEED", 42))
OUT, LOSS = os.environ.get("OUT", "ckpt"), os.environ.get("LOSS", "ce")
LR, EP = float(os.environ.get("LR", 3e-5)), float(os.environ.get("EP", 3))
# Routing cost of predicting column a when truth is row y: sending hard work to Go costs most.
COST = torch.tensor([[0, .3, 1], [.5, 0, .5], [3, 1.5, 0]])
NAMES = ["easy", "borderline", "hard"]
tok = AutoTokenizer.from_pretrained(BASE)
load = lambda f: Dataset.from_list([json.loads(l) for l in open(f)])
enc = lambda b: tok(b["text"], truncation=True, max_length=MAXLEN)
cols = ["text"]
AM, LM = {"easy": 0, "tie": 1, "hard": 2}, {"easy": 0, "medium": 1, "hard": 2}
soft = lambda b: {"soft": [[.5 * (c == y) + .25 * (c == AM[a]) + .25 * (c == LM[l]) for c in range(3)]
                           for y, a, l in zip(b["label"], b["arena"], b["luna"])]}
SMOKE = os.environ.get("SMOKE")
tr = load(os.environ.get("TRAIN", "train3.jsonl")).map(enc, batched=True, remove_columns=cols)
va = load("val2.jsonl").map(lambda b: {"has_g": [0.0] * len(b["text"]), "gp": [0.0] * len(b["text"])}, batched=True).map(enc, batched=True, remove_columns=cols)
JW, GW = float(os.environ.get("JW", 2)), float(os.environ.get("GW", 2))

if SMOKE:
    tr, va = tr.select(range(64)), va.select(range(64))
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=7)


class WTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        inputs.pop("weight"); inputs.pop("labels")
        ay, js, hj = inputs.pop("arena_y"), inputs.pop("jsoft"), inputs.pop("has_j")
        hg, gp = inputs.pop("has_g"), inputs.pop("gp")
        out = model(**inputs)
        lg = out.logits.float()
        ce = torch.nn.functional.cross_entropy(lg[:, :3], ay, reduction="none", label_smoothing=LS, ignore_index=-100)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(lg[:, 6], gp.float(), reduction="none")
        loss = (ce - JW * hj * (js * lg[:, 3:6].log_softmax(-1)).sum(-1) + GW * hg * bce).mean()
        return (loss, out) if return_outputs else loss


def metrics(p):
    pred, y = p.predictions.argmax(-1), p.label_ids
    lg = torch.tensor(p.predictions).float()
    pr = (lg[:, :3].softmax(-1) + lg[:, 3:6].softmax(-1)) / 2
    rk = lambda v: v.argsort().argsort().float() / len(v)
    pred, ph = pr.argmax(-1).numpy(), ((rk(pr[:, 2]) + rk(-lg[:, 6])) / 2).numpy()
    pos, neg = ph[y == 2], ph[y == 0]
    auc = float((pos[:, None] > neg[None]).mean() + .5 * (pos[:, None] == neg[None]).mean())
    f1 = [2 * ((pred == c) & (y == c)).sum() / max(1, (pred == c).sum() + (y == c).sum()) for c in range(3)]
    au = lambda h: float(((h[y == 2][:, None] > h[y == 0][None]).mean() + .5 * (h[y == 2][:, None] == h[y == 0][None]).mean()))
    extra = {"auc_arena_head": au(lg[:, :3].softmax(-1).numpy()[:, 2]), "auc_judge_head": au(lg[:, 3:6].softmax(-1).numpy()[:, 2]), "auc_gold_head": au(-lg[:, 6].numpy()), "auc_aj": au(pr.numpy()[:, 2])}
    return {**extra, "acc": float((pred == y).mean()), "macro_f1": float(np.mean(f1)), "auc_he": auc,
            "hard_recall": float((pred[y == 2] == 2).mean()),
            "hard_called_easy": float((pred[y == 2] == 0).mean()),
            "easy_precision": float((y[pred == 0] == 0).mean()) if (pred == 0).any() else 0.0}


args = TrainingArguments(
    OUT, num_train_epochs=EP, seed=SEED, learning_rate=LR, warmup_steps=0.06, weight_decay=0.01,
    per_device_train_batch_size=8, gradient_accumulation_steps=4, per_device_eval_batch_size=32,
    bf16=True, max_steps=4 if SMOKE else -1, train_sampling_strategy="group_by_length", eval_strategy="steps", save_strategy="steps", eval_steps=int(os.environ.get("ES", 600)), save_steps=int(os.environ.get("ES", 600)),
    save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="auc_he",
    logging_steps=20, report_to=[], dataloader_num_workers=2, remove_unused_columns=False)
t = WTrainer(model=model, args=args, train_dataset=tr.rename_column("label", "labels"),
             eval_dataset=va.rename_column("label", "labels"),
             data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
t.train(resume_from_checkpoint=bool(glob.glob(f"{OUT}/checkpoint-*")))
print("FINAL", json.dumps(t.evaluate()))
np.save(f"{OUT}/val_logits.npy", t.predict(va.rename_column("label", "labels")).predictions)
t.save_model(f"{OUT}-final")
tok.save_pretrained(f"{OUT}-final")
