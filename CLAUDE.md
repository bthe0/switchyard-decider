# Guide for AI coding agents

switchyard-decider is a ModernBERT-large classifier that scores how hard a coding-assistant request is, plus a small
HTTP server used by a local LLM router. See README.md and docs/ for background.

## Layout

- `switchyard_decider/serve.py`: pure numpy scoring functions (`request_text`, `hardness`, `p_solve`) and the server
  (`load_scorer`, `main`). torch and transformers are imported only inside functions that need them.
- `switchyard_decider/calib.py`, `switchyard_decider/predict.py`: calibration and logit export CLIs.
- `training/`: data assembly, teacher labelling, training, weight soup. Research scripts; they run from a data
  directory and read and write files in the current working directory. `training/retune/` holds the v1.2 retune
  (augmentation, fine-tune, probe refit, acceptance, latency); run it with the repo on `PYTHONPATH`.
- `benchmarks/`: decision-model benchmark, offline evaluation, latency. `benchmarks/holdout/holdout.jsonl` is the
  frozen v1.2 holdout. Never edit it, and never train or tune on it.
- `tests/`: numpy-only unit tests. CI installs only numpy, pytest and ruff, so tests must not import torch.
- `docs/`: architecture, serving, training, benchmarks, data, Hugging Face model card.

## Commands

```sh
pip install -e ".[dev]"              # add ,train for training scripts
ruff check .
pytest -q tests
switchyard-decider-serve <model_dir> --port <port>
python benchmarks/bench_speed.py <model_dir> <mps|cuda|cpu> [prompts.jsonl]
python benchmarks/evalp.py <logits.npy>...   # from the data directory
```

## Invariants

- Logit order is fixed: 0–2 arena easy/medium/hard, 3–5 judge easy/medium/hard, 6 gold. The gold logit is higher
  for easier requests, so hardness uses its negation. Changing either breaks the released weights.
- `calib.npz` has the keys `aj`, `g`, `h`, `out` (sorted arrays). From v1.1 on, it also has the ridge probe keys `ra`,
  `rb`, `rs`, `rw`. `calib.py` and `training/retune/refit_v12.py` write it, and `serve.py` reads it; keep them in sync.
- API contract: `POST /v1/systemone` with `{"state": {"opening_task": ..., "latest_follow_up": ...}}` returns
  `{"answers": {"solve": {"noul": p}}}`. The input is `opening_task[:20000]` and `latest_follow_up` joined by a blank
  line, cut to 20,000 characters, tokenized, then cut to the first 384 and last 384 tokens if longer than 768
  (`encode`/`head_tail` in `serve.py`). v1.2 weights depend on this cut. Output values must not change for the same
  weights and input.
- Never commit secrets or API keys; scripts read `OPENROUTER_API_KEY` from the environment.
- Datasets, logits and weights are not in git (`.gitignore` covers `*.jsonl`, `*.npy`, `*.npz`, `*.safetensors`). The
  one exception is the frozen holdout, `benchmarks/holdout/holdout.jsonl`.
- Do not bind test servers to a port another service may use; pick a free high port.

## Releasing weights

1. Build the model directory: `model.safetensors` (fp16), `config.json`, `tokenizer.json`, `tokenizer_config.json`,
   `calib.npz`.
2. Hugging Face: `hf upload bthe0/switchyard-decider <model_dir> .` and upload `docs/MODEL_CARD.md` as `README.md`.
3. GitHub: `gh release create v<version> <model_dir>/* -R bthe0/switchyard-decider`.
4. Bump `version` in `pyproject.toml` and `switchyard_decider/__init__.py`, and add a CHANGELOG.md entry.
