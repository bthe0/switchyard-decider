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
local LLM router uses the score to send a request either to a cheaper mid-tier model or to a frontier model. It
answers in about 25 ms (p50, Apple-silicon MPS) and reaches AUC 0.895 on held-out easy vs hard prompts.

- Weights: https://huggingface.co/bthe0/switchyard-decider (primary) and the GitHub release
  https://github.com/bthe0/switchyard-decider/releases/tag/v1.1
- Code: https://github.com/bthe0/switchyard-decider

Files: `model.safetensors` (about 792 MB, fp16), `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`calib.npz`.

```sh
hf download bthe0/switchyard-decider --local-dir switchyard-decider
# or
gh release download v1.1 -R bthe0/switchyard-decider -D switchyard-decider
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
with torch.no_grad():
    lg = model(**tok(text, truncation=True, max_length=768, return_tensors="pt")).logits[0].float()
pr = (lg[:3].softmax(-1) + lg[3:6].softmax(-1)) / 2
hardness = (np.searchsorted(calib["aj"], pr[2].item()) + np.searchsorted(calib["g"], -lg[6].item())) / 2 / len(calib["aj"])
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
the negated gold logit. `calib.npz` (keys `aj`, `g`, `h`, `out`) holds the reference arrays and maps hardness onto a
target output distribution so that a router's existing cutoffs keep routing the same share of traffic.

## Training data

About 153k prompts from WildChat-1M (ODC-BY-1.0), LMArena arena-human-preference-140k (CC-BY-4.0), OpenAssistant
oasst2 (Apache-2.0), Magicoder-Evol-Instruct-110K (Apache-2.0), CodeFeedback-Filtered-Instruction (Apache-2.0),
SWE-bench (MIT), SWE-Gym (MIT) and SWE-rebench (CC-BY-4.0). Labels came from arena battle outcomes, LLM judges
(Anthropic Claude Haiku, OpenAI GPT-6-luna decisions) and OpenRouter decision models (`nace-ai/drex-max`,
`perplexity/pplx-decider`).

Training: lr 3e-5, max length 768, bf16, single NVIDIA RTX 5070 Ti, best checkpoint at step 3,600, followed by a
greedy weight soup.

## Evaluation

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

Apple M4 Max (48 GB), PyTorch, single request, batch 1. Forward pass only at fixed lengths; "real prompts" includes
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
- Only the first 3,000 characters of the opening task plus the latest follow-up are scored, truncated to 768 tokens.
- Labels are noisy teacher outputs; the model ranks difficulty, it does not verify that a model can solve a task.
- Evaluation numbers are slightly optimistic because the checkpoint was selected on the validation set.

## License and usage risk

Code and weights are released under Apache-2.0, matching the ModernBERT base model. The training labels are outputs of
commercial models. Some providers' terms restrict using outputs to build competing models. This model is a
binary-ish routing classifier, not a general model, but check the relevant terms before any commercial use. The
WildChat (ODC-BY) and CC-BY sources require attribution, which the list above provides.
