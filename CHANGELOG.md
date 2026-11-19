# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.2.0] - 2026-11-19

Fixes v1.1's length shortcut. On a frozen 200-row holdout, hard-vs-easy AUC rises from 0.8396 to 0.9794 (paired
bootstrap 95% lower bound of the gain +0.0906). Matched pairs ordered correctly go from 13/20 to 19/20. val2 AUC moves
from 0.8952 to 0.8966 and fresh100 AUC from 0.9908 to 0.9956.

### Added

- `head_tail` and `encode` in `switchyard_decider/serve.py`, with unit tests: inputs over 768 tokens keep the first 384
  and last 384 tokens.
- `benchmarks/holdout/`: the frozen 200-row holdout (sha256 `967dc16e…c61357`) and the deterministic generator that
  reproduces it.
- `training/retune/`: augmentation builder, v1.2 fine-tune script, probe refit, holdout QA and leakage checks,
  acceptance scoring with paired bootstrap, and an end-to-end latency harness.
- `TRUNC` option in `training/train3.py` (`ht` by default, `head` reproduces v1.0 and v1.1).

- `benchmarks/bench_llm.py`: compares the served model with zero-shot OpenRouter chat LLMs and OpenRouter decision
  models on the same validation sample. Reports AUC with bootstrap CI, routing accuracy, latency, tokens and cost
  per decision, and has a spend cap. Results are in `benchmarks/results/openrouter_compare.json`, the chart is
  `docs/pareto.png`, and the write-up is in docs/benchmarks.md. These numbers were measured on v1.1.

### Changed

- Weights: v1.2 is fine-tuned from v1.1 for 1 epoch at lr 1e-5 on 20k replay rows plus 4,653 label-preserving
  augmentation rows, with the head+tail cut. The ridge probe was refit. Training took about 45 minutes on one RTX
  5070 Ti.
- The server reads up to 20,000 characters of `opening_task` (was 3,000) and applies the head+tail cut instead of
  keeping the first 768 tokens. `predict.py` and `training/embed.py` use the same `encode`. With v1.1 weights, scores
  for inputs over 768 tokens change. v1.2 weights need this cut.
- Docs and model card report v1.2 results. Latency is now measured end to end on long holdout texts (62.8 ms p50 for
  v1.2, 64.0 ms for v1.1), so it is not comparable with the earlier short-prompt figure of about 25 ms.

## [1.1.0] - 2026-11-19

### Added

- Ridge probe on the frozen CLS+mean embedding, rank-blended with the head score (weight 4). Validation AUC 0.886 → 0.895
  (5-fold CV of the blend 0.895 vs 0.886), measured end to end through the server. No measurable latency cost.
  Stored in `calib.npz` (`ra`, `rb`, `rs`, `rw`); older calib files without them keep the 1.0 behaviour.
- `training/embed.py` and `training/fit_head.py` to rebuild the probe.

## [1.0.0] - 2026-11-19

### Added

- ModernBERT-large classifier with arena, LLM-judge and gold heads (validation AUC 0.886).
- HTTP server `switchyard-decider-serve` with `POST /v1/systemone` returning a calibrated mid-tier solve probability.
- Calibration (`calib.npz`) that maps hardness onto a target score distribution.
- Training pipeline, decision-model benchmark, offline evaluation and latency benchmark scripts.
- Installable package, unit tests, CI and documentation.
- Weights on Hugging Face (`bthe0/switchyard-decider`) and the GitHub release v1.0.
