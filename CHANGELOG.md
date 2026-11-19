# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

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
