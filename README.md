# switchyard-decider

A small local classifier that decides whether a coding-assistant request needs a frontier model or can be handled
by a cheaper mid-tier model. It replaces an LLM-based decider in a local model router and answers in ~25 ms (p50,
Apple-silicon MPS) instead of ~700 ms.

The model is `answerdotai/ModernBERT-large` fine-tuned with seven output logits:

| logits | target | loss |
|---|---|---|
| 0–2 | arena-derived easy / medium / hard label | cross-entropy |
| 3–5 | LLM-judge soft label (easy / medium / hard) | soft cross-entropy |
| 6 | "gold" hardness from OpenRouter decision models, percentile-calibrated per teacher | BCE |

At inference the hardness score is the mean of two percentiles (validation set as reference): P(hard) averaged over
the arena and judge heads, and the negated gold logit. `serve.py` maps that percentile onto a target output
distribution stored in `calib.npz`, so a router's existing probability cutoffs keep routing the same share of traffic
while the ranking improves.

## Results

Validation set: 1,994 held-out prompts (easy vs hard pairs; medium excluded from AUC).

| model | AUC (labels) | AUC (arena) | AUC (coding) | hard → frontier @40% | easy → mid-tier @40% |
|---|---|---|---|---|---|
| **this model (step 3600)** | **0.886** | **0.769** | **0.929** | **0.693** | **0.862** |
| same recipe without gold head | 0.859 | | | | |
| previous LLM decider (0.8B, MLX 4-bit) | 0.687 | | | 0.538 | 0.712 |

On a fixed 300-prompt subset used to benchmark ~40 OpenRouter decision models, this model scores AUC 0.904. The best
teachers scored 0.879 (`nace-ai/drex-max-finance`), 0.873 (`nace-ai/drex-max`) and 0.871
(`perplexity/pplx-decider-v1.3`). Combining labels from several teachers with independent errors beats any single
teacher.

Caveat: the checkpoint was chosen on this validation set, so these numbers are slightly optimistic.

Training notes:
- 153k rows, 2 epochs planned, lr 3e-5, max length 768, bf16, single RTX 5070 Ti.
- Stopped early at step ~4,450 of 9,568. Eval AUC peaked at step 3,600 (0.886) and was flat to declining after that.
- Weight-averaging ("model soup") of five earlier variants gave +0.009 AUC over the best single run. The gold-label
  head gave +0.027.

## Files

- `serve.py`: HTTP server. `POST /v1/systemone` with `{"state": {"opening_task": ..., "latest_follow_up": ...}}`
  returns `{"answers": {"solve": {"noul": p}}}`, where `p` is the probability the request is mid-tier-solvable.
  Run it with `python serve.py <model_dir> [port]`.
- `calib.py`: builds `calib.npz` from validation logits and an optional target score distribution.
- `train3.py`: multi-head training (env: `BASE`, `LR`, `EP`, `MAXLEN`, `LS`, `SEED`, `JW`, `GW`, `OUT`).
- `make_splits2.py`, `gold_build.py`, `gold_label.py`, `build_train3.py`: data assembly and teacher labelling.
- `bench_dec.py`: benchmarks OpenRouter decision models on the validation subset (needs `OPENROUTER_API_KEY`).
- `predict.py`, `evalp.py`, `soup.py`: inference, evaluation, and greedy weight soup.

Datasets are not included. The weights (fp16 safetensors + tokenizer + `calib.npz`) are attached to the GitHub release.

## Data sources

Prompts came from public datasets, used under their licenses:
- WildChat-1M (ODC-BY-1.0)
- LMArena arena-human-preference-140k (CC-BY-4.0)
- OpenAssistant oasst2 (Apache-2.0)
- Magicoder-Evol-Instruct-110K (Apache-2.0)
- CodeFeedback-Filtered-Instruction (Apache-2.0)
- SWE-bench (MIT)
- SWE-Gym (MIT)
- SWE-rebench (CC-BY-4.0)

Labels were produced by LLM judges (Anthropic Claude Haiku, OpenAI GPT-6-luna decisions) and by OpenRouter decision
models (`nace-ai/drex-max`, `perplexity/pplx-decider`).

## License and usage risk

Code and weights are released under Apache-2.0, matching the ModernBERT base model. The training labels are outputs of
commercial models. Some providers' terms restrict using outputs to build competing models. This model is a
binary-ish routing classifier, not a general model, but check the relevant terms before any commercial use. The
WildChat (ODC-BY) and CC-BY sources require attribution, which the list above provides.
