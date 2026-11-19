# Guide for AI coding agents

switchyard-decider is a ModernBERT-large classifier that scores how hard a coding-assistant request is, plus a small
HTTP server used by a local LLM router. See README.md and docs/ for background.

## Layout

- `switchyard_decider/serve.py`: pure numpy scoring functions (`request_text`, `hardness`, `p_solve`) and the server
  (`load_scorer`, `main`). torch and transformers are imported only inside functions that need them.
- `switchyard_decider/calib.py`, `switchyard_decider/predict.py`: calibration and logit export CLIs.
- `training/`: data assembly, teacher labelling, training, weight soup. Research scripts; they run from a data
  directory and read and write files in the current working directory.
- `benchmarks/`: decision-model benchmark, offline evaluation, latency.
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
- `calib.npz` has exactly the keys `aj`, `g`, `h`, `out` (sorted arrays). `calib.py` writes it and `serve.py` reads it;
  keep both in sync.
- API contract: `POST /v1/systemone` with `{"state": {"opening_task": ..., "latest_follow_up": ...}}` returns
  `{"answers": {"solve": {"noul": p}}}`. The input is `opening_task[:3000]` and `latest_follow_up` joined by a blank
  line, truncated to 768 tokens. Output values must not change for the same weights and input.
- Never commit secrets or API keys; scripts read `OPENROUTER_API_KEY` from the environment.
- Datasets, logits and weights are not in git (`.gitignore` covers `*.jsonl`, `*.npy`, `*.npz`, `*.safetensors`).
- Do not bind test servers to a port another service may use; pick a free high port.

## Releasing weights

1. Build the model directory: `model.safetensors` (fp16), `config.json`, `tokenizer.json`, `tokenizer_config.json`,
   `calib.npz`.
2. Hugging Face: `hf upload bthe0/switchyard-decider <model_dir> .` and upload `docs/MODEL_CARD.md` as `README.md`.
3. GitHub: `gh release create v<version> <model_dir>/* -R bthe0/switchyard-decider`.
4. Bump `version` in `pyproject.toml` and `switchyard_decider/__init__.py`, and add a CHANGELOG.md entry.
