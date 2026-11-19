# switchyard-decider

[![CI](https://github.com/bthe0/switchyard-decider/actions/workflows/ci.yml/badge.svg)](https://github.com/bthe0/switchyard-decider/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20model-bthe0%2Fswitchyard--decider-yellow)](https://huggingface.co/bthe0/switchyard-decider)

A small local classifier that decides whether a coding-assistant request needs a frontier model or can be handled by
a cheaper mid-tier model. It is a fine-tuned `answerdotai/ModernBERT-large` that replaces an LLM-based decider in a
local LLM router. On Apple M4 Max MPS it answers long inputs in about 63 ms p50 end to end. Typical short prompts
take about 25 ms (a v1.1 measurement on the same architecture). The LLM decider it replaces took about 700 ms. A small
HTTP server returns a calibrated probability that the request is solvable by the mid-tier model.

v1.2 fixes the length shortcut in v1.1, which treated any long paste as hard. On a frozen holdout built to test that
shortcut, AUC goes from 0.8396 to 0.9794.

## Weights

Weights are not stored in git. Download them from either source:

- Hugging Face (primary): https://huggingface.co/bthe0/switchyard-decider

  ```sh
  hf download bthe0/switchyard-decider --local-dir switchyard-decider
  ```

- GitHub release v1.2: https://github.com/bthe0/switchyard-decider/releases/tag/v1.2

  ```sh
  gh release download v1.2 -R bthe0/switchyard-decider -D switchyard-decider
  ```

Assets: `model.safetensors` (about 792 MB, fp16), `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`calib.npz`.

The v1.2 weights need the v1.2 code (package 1.2.0), which keeps the first 384 and last 384 tokens of inputs longer
than 768 tokens. The v1.1 server keeps only the first 768 tokens and gives wrong scores for long inputs with v1.2
weights.

## Results

### v1.2 (served)

| metric | v1.1 | v1.2 |
|---|---|---|
| holdout hard-vs-easy AUC (200 rows) | 0.8396 | **0.9794** |
| holdout matched pairs ordered correctly | 13/20 | 19/20 |
| holdout long_easy on the wrong side of the cutoff | 30/35 | 13/35 |
| holdout pair_easy on the wrong side | 16/20 | 8/20 |
| holdout short_hard on the wrong side | 2/35 | 0/35 |
| holdout controls (short_easy, long_hard) on the wrong side | 0 | 0 |
| val2 AUC | 0.8952 | 0.8966 |
| fresh100 AUC | 0.9908 | 0.9956 |
| dev-aug AUC | 0.7312 | 0.9576 |
| p50 latency, end to end, Apple M4 Max MPS fp16, batch 1, holdout texts | 64.0 ms | 62.8 ms (p90 64.9 ms) |

Holdout AUC difference: +0.1398. A paired bootstrap with 2,000 resamples puts the 95% lower bound at +0.0906. The
holdout (`benchmarks/holdout/`) has 200 frozen rows:

| subset | rows | what it tests |
|---|---|---|
| short_easy, short_hard | 35 + 35 | short asks (controls) |
| long_easy, long_hard | 35 + 35 | the same kinds of asks after long logs, JSON or code pastes |
| pair_easy, pair_hard | 20 + 20 | the same paste with an easy ask and with a hard ask |
| borderline | 20 | medium asks (excluded from AUC) |

The holdout was never used for training or tuning. It has 0 exact matches and 0 near-duplicates (8-gram overlap
≥ 50%) against the training, augmentation, dev-aug and fresh100 sets. Length alone scores AUC 0.5065 on it. The
checkpoint was selected by a rule recorded before any holdout metric existed: highest dev-aug AUC among runs with
val2 AUC ≥ 0.8902. Training took about 45 minutes on one RTX 5070 Ti, about $0.25 rented. Details are in
[docs/benchmarks.md](docs/benchmarks.md#holdout-v12) and [docs/training.md](docs/training.md#v12-retune).

Latency: the v1.2 harness (`training/retune/bench_latency.py`) times the served path end to end. That covers the
character cut, tokenization of long inputs, the head+tail cut and the ridge probe, on long holdout texts. The older
figures (about 25 ms p50 through the server) came from short validation prompts (median 80 tokens), so the two
measurements are not comparable.

### Validation history (v1.0, v1.1)

Validation set: 1,994 held-out prompts (easy vs hard pairs; medium excluded from AUC).

| model | AUC (labels) | AUC (arena) | AUC (coding) | hard → frontier @40% | easy → mid-tier @40% |
|---|---|---|---|---|---|
| **+ ridge probe on embedding (v1.1, served)** | **0.895** | | | | |
| this model (step 3600, head only, v1.0) | 0.886 | 0.769 | 0.929 | 0.693 | 0.862 |
| previous LLM decider (0.8B, MLX 4-bit) | 0.687 | | | 0.538 | 0.712 |

OpenRouter comparison (measured on v1.1, not rerun for v1.2): on a 120-prompt easy/hard sample, v1.1 reaches AUC 0.925 (95% CI 0.877-0.964) in 26 ms p50 locally with $0 API cost.
The best OpenRouter decision model measured for latency (drex-max-finance) reaches 0.917 in 494 ms at $18 per 1M
decisions. The best zero-shot chat LLM (gemini-3.1-flash-lite) reaches 0.841 in 1,226 ms at $262 per 1M decisions.

The v1.0 checkpoint was chosen on this validation set, so these numbers are slightly optimistic. Full tables, the
comparison with about 40 OpenRouter decision models and 10 chat LLMs, a Pareto chart and latency figures are in
[docs/benchmarks.md](docs/benchmarks.md#comparison-with-openrouter-models).

Hardware: training on one NVIDIA RTX 5070 Ti (16 GB). Latency on an Apple M4 Max (48 GB) with PyTorch MPS.

## Limitations

- Long pastes still pull some easy asks toward "hard": 13/35 long_easy and 8/20 pair_easy holdout rows remain on the
  wrong side of the cutoff.
- The holdout is synthetic. Its asks were written by an LLM and assembled by deterministic generator scripts. A TF-IDF
  classifier separates its classes at CV AUC 0.597, a borderline amount of lexical leakage.
- dev-aug is built the same way as the training augmentation, so the dev-aug gain is optimistic.
- Inputs over 768 tokens lose their middle. Only the first 384 and last 384 tokens are scored.
- The 40 ms latency goal is not met on long inputs by either v1.1 or v1.2; batch-1 MPS inference is bound by kernel
  launch overhead.
- English only, tuned on coding-assistant and chat prompts. Labels are noisy teacher outputs.

## Quickstart

```sh
pip install -e .
hf download bthe0/switchyard-decider --local-dir switchyard-decider
switchyard-decider-serve switchyard-decider --port 4126
```

```sh
curl -s -X POST http://127.0.0.1:4126/v1/systemone \
  -H 'content-type: application/json' \
  -d '{"state": {"opening_task": "Rename the variable foo to bar in utils.py", "latest_follow_up": ""}}'
# {"answers": {"solve": {"noul": 0.84}}}
```

`noul` is the probability that the request can be handled by the mid-tier model. See
[docs/serving.md](docs/serving.md) for the API contract and running it as a service.

## Repository layout

```
switchyard_decider/   installable package
  serve.py            HTTP server and pure scoring functions
  calib.py            builds calib.npz from validation logits
  predict.py          writes raw logits for a validation file
training/             data assembly, teacher labelling, training, weight soup
  retune/             v1.2 augmentation, fine-tune, probe refit, acceptance and latency scripts
benchmarks/           decision-model benchmark, offline evaluation, latency
  holdout/            frozen 200-row v1.2 holdout and its deterministic generator
tests/                unit tests for the scoring functions (numpy only)
docs/                 architecture, serving, training, benchmarks, data, model card
```

## Documentation

- [Architecture](docs/architecture.md): model, heads, hardness scoring and calibration
- [Serving](docs/serving.md): install, run, HTTP API, running as a service
- [Training](docs/training.md): pipeline order, script inputs and outputs, hardware
- [Benchmarks](docs/benchmarks.md): accuracy, teacher comparison, speed, OpenRouter comparison
- [Data](docs/data.md): sources, licenses and labelling
- [Model card](docs/MODEL_CARD.md): the Hugging Face model card

## License

Code and weights are released under the [Apache-2.0](LICENSE) license, matching the ModernBERT base model. The
training labels are outputs of commercial models, and some providers' terms restrict using outputs to build competing
models; read [docs/data.md](docs/data.md#license-and-usage-risk) before any commercial use.
