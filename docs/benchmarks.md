# Benchmarks

## Accuracy

Validation set: 1,994 held-out prompts (easy vs hard pairs; medium excluded from AUC).

| model | AUC (labels) | AUC (arena) | AUC (coding) | hard → frontier @40% | easy → mid-tier @40% |
|---|---|---|---|---|---|
| **+ ridge probe on embedding (v1.1, served)** | **0.895** | | | | |
| this model (step 3600, head only, v1.0) | 0.886 | 0.769 | 0.929 | 0.693 | 0.862 |
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

Reproduce with `python benchmarks/bench_speed.py <model_dir> <mps|cuda|cpu> [prompts.jsonl]`, which reports forward
latency at fixed token lengths and, if a prompts file is given, end-to-end latency on real prompts.
