# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-19

### Added

- ModernBERT-large classifier with arena, LLM-judge and gold heads (validation AUC 0.886).
- HTTP server `switchyard-decider-serve` with `POST /v1/systemone` returning a calibrated mid-tier solve probability.
- Calibration (`calib.npz`) that maps hardness onto a target score distribution.
- Training pipeline, decision-model benchmark, offline evaluation and latency benchmark scripts.
- Installable package, unit tests, CI and documentation.
- Weights on Hugging Face (`bthe0/switchyard-decider`) and the GitHub release v1.0.
