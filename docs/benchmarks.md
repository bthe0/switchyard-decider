# Benchmarks

## Accuracy

Validation set: 1,994 held-out prompts (easy vs hard pairs; medium excluded from AUC).

| model | AUC (labels) | AUC (arena) | AUC (coding) | hard → frontier @40% | easy → mid-tier @40% |
|---|---|---|---|---|---|
| **this model (step 3600)** | **0.886** | **0.769** | **0.929** | **0.693** | **0.862** |
| same recipe without gold head | 0.859 | | | | |
| previous LLM decider (0.8B, MLX 4-bit) | 0.687 | | | 0.538 | 0.712 |

Columns:

- AUC (labels): easy vs hard on the combined validation label.
- AUC (arena): easy vs hard on the arena-derived label only.
- AUC (coding): easy vs hard restricted to prompts in the coding domain.
- hard → frontier @40% and easy → mid-tier @40%: when the hardest 40% of requests go to the frontier model, the share
  of hard requests sent to the frontier model and the share of easy requests kept on the mid-tier model.

Caveat: the checkpoint was chosen on this validation set, so these numbers are slightly optimistic.

Reproduce with `switchyard_decider/predict.py` to write logits, then `benchmarks/evalp.py <logits.npy>...` from the
data directory (needs `val.jsonl` with `label`, `arena` and `domain` fields).

## Teacher comparison

On a fixed 300-prompt subset used to benchmark about 40 OpenRouter decision models, this model scores AUC 0.904. The
best teachers scored 0.879 (`nace-ai/drex-max-finance`), 0.873 (`nace-ai/drex-max`) and 0.871
(`perplexity/pplx-decider-v1.3`). Combining labels from several teachers with independent errors beats any single
teacher.

Reproduce with `benchmarks/bench_dec.py [model...]` (needs `OPENROUTER_API_KEY`; `N` sets the subset size, default
300; `CAP` stops when account usage reaches that many dollars). Results are cached per model in `dec_bench/`.

## Speed

<!-- SPEED -->

Reproduce with `python benchmarks/bench_speed.py <model_dir> <mps|cuda|cpu> [prompts.jsonl]`, which reports forward
latency at fixed token lengths and, if a prompts file is given, end-to-end latency on real prompts.
