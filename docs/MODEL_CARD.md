---
license: apache-2.0
base_model: answerdotai/ModernBERT-large
pipeline_tag: text-classification
library_name: transformers
language:
  - en
tags:
  - router
  - llm-routing
  - modernbert
  - difficulty-estimation
---

# switchyard-decider

## Summary

switchyard-decider is `answerdotai/ModernBERT-large` fine-tuned to score how hard a coding-assistant request is. A
local LLM router uses the score to send a request either to a cheaper mid-tier model or to a frontier model. v1.2
reaches AUC 0.9794 on a frozen 200-row holdout built to catch length shortcuts (v1.1: 0.8396) and 0.8966 on the
validation set val2 (v1.1: 0.8952), at 62.8 ms p50 end to end on Apple M4 Max MPS.

v1.2 needs the head+tail input cut from the v1.2 code (`switchyard_decider/serve.py`, package version 1.2.0). Do not
serve these weights with the v1.1 server, which keeps only the first 768 tokens.

- Weights: https://huggingface.co/bthe0/switchyard-decider (primary) and the GitHub release
  https://github.com/bthe0/switchyard-decider/releases/tag/v1.2
- Code: https://github.com/bthe0/switchyard-decider

Files: `model.safetensors` (about 792 MB, fp16), `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`calib.npz`.

```sh
hf download bthe0/switchyard-decider --local-dir switchyard-decider
# or
gh release download v1.2 -R bthe0/switchyard-decider -D switchyard-decider
```

## Intended use

Routing single requests to coding assistants by estimated difficulty: lower scores go to a frontier model, higher
scores to a mid-tier model. It is not a general-purpose classifier and has not been evaluated outside English
coding-assistant and chat prompts.

## How to use

With transformers and numpy:

```python
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

d = "switchyard-decider"
tok = AutoTokenizer.from_pretrained(d)
model = AutoModelForSequenceClassification.from_pretrained(d, dtype=torch.float32).eval()
calib = np.load(f"{d}/calib.npz")

text = "Rename the variable foo to bar in utils.py"
ids = tok(text[:20000])["input_ids"]
if len(ids) > 768:  # head+tail: keep the first 384 and last 384 tokens
    ids = ids[:384] + ids[-384:]
with torch.no_grad():
    out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
lg, hid = out.logits[0].float(), out.hidden_states[-1][0].float()
pr = (lg[:3].softmax(-1) + lg[3:6].softmax(-1)) / 2
h = (np.searchsorted(calib["aj"], pr[2].item()) + np.searchsorted(calib["g"], -lg[6].item())) / 2 / len(calib["aj"])
emb = torch.cat([hid[0], hid.mean(0)]).numpy()  # CLS + mean of the final hidden states
r = np.searchsorted(calib["rs"], emb @ calib["ra"] + calib["rb"]) / len(calib["rs"])
hardness = (h + calib["rw"] * r) / (1 + calib["rw"])
q = np.searchsorted(calib["h"], hardness) / len(calib["h"])
p_solve = float(np.quantile(calib["out"], 1 - q))  # probability the mid-tier model can handle it
```

Or run the HTTP server from the code repository:

```sh
pip install -e .
switchyard-decider-serve switchyard-decider --port 4126
curl -s -X POST http://127.0.0.1:4126/v1/systemone \
  -d '{"state": {"opening_task": "Rename the variable foo to bar in utils.py", "latest_follow_up": ""}}'
# {"answers": {"solve": {"noul": 0.84}}}
```

## Outputs

Seven logits:

| logits | meaning |
|---|---|
| 0–2 | arena-derived easy / medium / hard |
| 3–5 | LLM-judge easy / medium / hard |
| 6 | gold score from decision-model teachers; higher means easier |

The hardness score is the mean of two validation-set percentiles: P(hard) averaged over the arena and judge heads, and
the negated gold logit. That is rank-blended with a ridge probe on the CLS+mean embedding (`ra`, `rb`, `rs`, weight
`rw`). `calib.npz` holds these reference arrays (`aj`, `g`, `h`, `out`, `ra`, `rb`, `rs`, `rw`) and maps hardness
onto a target output distribution so that a router's existing cutoffs keep routing the same share of traffic.

## Training data

About 153k prompts from WildChat-1M (ODC-BY-1.0), LMArena arena-human-preference-140k (CC-BY-4.0), OpenAssistant
oasst2 (Apache-2.0), Magicoder-Evol-Instruct-110K (Apache-2.0), CodeFeedback-Filtered-Instruction (Apache-2.0),
SWE-bench (MIT), SWE-Gym (MIT) and SWE-rebench (CC-BY-4.0). Labels came from arena battle outcomes, LLM judges
(Anthropic Claude Haiku, OpenAI GPT-6-luna decisions) and OpenRouter decision models (`nace-ai/drex-max`,
`perplexity/pplx-decider`).

Training (v1.0): lr 3e-5, max length 768, bf16, single NVIDIA RTX 5070 Ti, best checkpoint at step 3,600, followed by
a greedy weight soup. v1.1 added the ridge probe.

v1.2 retune, from the v1.1 weights: 1 epoch at lr 1e-5 on 20k replay rows of the original training set plus 4,653
label-preserving augmentation rows. The augmentation covers easy asks wrapped in long logs, JSON or code pastes; terse
hard questions; and matched pairs that put an easy ask and a hard ask after the same paste. Inputs over 768 tokens
keep the first 384 and last 384 tokens. The ridge probe was refit with no augmentation weight. Training took about
45 minutes on one RTX 5070 Ti (about $0.25 rented).

The selection rule was fixed before any holdout metric was computed: take the highest dev-aug AUC among runs with
val2 AUC ≥ 0.8902. Candidates scored (val2 / dev-aug): r5 0.8936 / 0.9463, r10 0.8968 / 0.9483, r20 0.8987 / 0.9505,
r10ht 0.8966 / 0.9576 (chosen). Probe refits that weighted the augmentation rows failed the val2 gate.

## Evaluation

### v1.2 against v1.1

The holdout is 200 frozen rows: 35 each of short_easy, short_hard, long_easy and long_hard, 20 matched pairs (same
paste, one easy ask and one hard ask), and 20 borderline rows. Its sha256 is
`967dc16e4d61e90ed14a1f4c048a14be285264f8193a1b14b211a87ed6c61357`. It was never used for training or tuning. It has
0 exact matches and 0 near-duplicates (8-gram overlap ≥ 50%) against the training, augmentation, dev-aug and fresh100
sets. Length alone scores AUC 0.5065 on it, so length does not separate its classes.

| metric | v1.1 | v1.2 |
|---|---|---|
| holdout hard-vs-easy AUC | 0.8396 | **0.9794** |
| matched pairs ordered correctly | 13/20 | 19/20 |
| long_easy on the wrong side of the cutoff | 30/35 | 13/35 |
| pair_easy on the wrong side | 16/20 | 8/20 |
| short_hard on the wrong side | 2/35 | 0/35 |
| short_easy, long_hard on the wrong side (controls) | 0 | 0 |
| val2 AUC | 0.8952 | 0.8966 |
| fresh100 AUC | 0.9908 | 0.9956 |
| dev-aug AUC | 0.7312 | 0.9576 |

Holdout AUC difference +0.1398. Paired bootstrap with 2,000 resamples gives a 95% lower bound of +0.0906.

### Validation history (v1.0, v1.1)

Validation set: 1,994 held-out prompts (easy vs hard pairs; medium excluded from AUC).

| model | AUC (labels) | AUC (arena) | AUC (coding) | hard → frontier @40% | easy → mid-tier @40% |
|---|---|---|---|---|---|
| **+ ridge probe on embedding (v1.1, served)** | **0.895** | | | | |
| this model (step 3600, head only, v1.0) | 0.886 | 0.769 | 0.929 | 0.693 | 0.862 |
| same recipe without gold head | 0.859 | | | | |
| previous LLM decider (0.8B, MLX 4-bit) | 0.687 | | | 0.538 | 0.712 |

On a fixed 300-prompt subset used to benchmark about 40 OpenRouter decision models, this model scores AUC 0.904. The
best teachers scored 0.879 (`nace-ai/drex-max-finance`), 0.873 (`nace-ai/drex-max`) and 0.871
(`perplexity/pplx-decider-v1.3`).

The checkpoint was chosen on this validation set, so these numbers are slightly optimistic.

## Speed

v1.2 measurement: served path end to end (20,000-character cut, tokenization, head+tail, forward pass, ridge probe),
batch 1, fp16, Apple M4 Max MPS, on the 200 holdout texts. Most of these texts are long. Same harness for both models:

| model | p50 | p90 |
|---|---|---|
| v1.1 | 64.0 ms | |
| v1.2 | 62.8 ms | 64.9 ms |

The earlier figures below come from a forward-pass harness on mostly short validation prompts (median 80 tokens).
They are not comparable with the table above, which includes tokenization of long inputs. Neither model meets a
40 ms goal on long inputs; batch-1 MPS inference is bound by kernel launch overhead.

Earlier measurement (v1.1): Apple M4 Max (48 GB), PyTorch, single request, batch 1. Forward pass only at fixed lengths; "real prompts" includes
tokenization on the 1,994 validation prompts (median 80 tokens). The ridge probe adds one 2048-wide dot product.

| tokens | MPS fp16 p50 | MPS fp16 p95 | CPU fp32 p50 | CPU fp32 p95 |
|---|---|---|---|---|
| 32 | 11.4 ms | 11.7 ms | 49.4 ms | 55.3 ms |
| 128 | 17.7 ms | 18.1 ms | 125.1 ms | 141.7 ms |
| 256 | 27.6 ms | 29.3 ms | 150.3 ms | 161.1 ms |
| 512 | 107.8 ms | 320.3 ms | 294.1 ms | 317.5 ms |
| 768 | 145.4 ms | 161.6 ms | 473.3 ms | 522.6 ms |
| real prompts | 30.2 ms (p99 151 ms) | 107.5 ms | 78.7 ms (p99 320 ms, 200 prompts) | 219.0 ms |

Through the HTTP server on MPS (500 prompts): p50 23 ms, p95 74 ms, p99 97 ms. Model load 0.7 s, resident memory
about 560 MB. The previous LLM decider took about 720 ms per decision.

## Limitations

- English only; tuned on coding-assistant and chat prompts.
- The model scores the first 20,000 characters of the opening task plus the latest follow-up. Inputs over 768 tokens
  keep only the first 384 and last 384 tokens, so text in the middle of a long paste is not seen.
- Long pastes still bias v1.2 toward "hard": 13/35 long_easy and 8/20 pair_easy holdout rows land on the wrong side of
  the cutoff.
- The holdout is synthetic. Its asks were written by an LLM and assembled by deterministic generator scripts. A TF-IDF
  classifier separates its classes at CV AUC 0.597, a borderline amount of lexical leakage.
- dev-aug is built the same way as the training augmentation, so the dev-aug gain is optimistic.
- The v1.2 weights need the head+tail serve path (v1.2 code). The v1.1 `serve.py` truncates to the first 768 tokens.
- Labels are noisy teacher outputs; the model ranks difficulty, it does not verify that a model can solve a task.
- Evaluation numbers are slightly optimistic because the checkpoint was selected on the validation set.

## License and usage risk

Code and weights are released under Apache-2.0, matching the ModernBERT base model. The training labels are outputs of
commercial models. Some providers' terms restrict using outputs to build competing models. This model is a
binary-ish routing classifier, not a general model, but check the relevant terms before any commercial use. The
WildChat (ODC-BY) and CC-BY sources require attribution, which the list above provides.
