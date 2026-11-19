# switchyard-decider

[![CI](https://github.com/bthe0/switchyard-decider/actions/workflows/ci.yml/badge.svg)](https://github.com/bthe0/switchyard-decider/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20model-bthe0%2Fswitchyard--decider-yellow)](https://huggingface.co/bthe0/switchyard-decider)

A small local classifier that decides whether a coding-assistant request needs a frontier model or can be handled by
a cheaper mid-tier model. It is a fine-tuned `answerdotai/ModernBERT-large` that replaces an LLM-based decider in a
local LLM router and answers in about 25 ms (p50, Apple-silicon MPS) instead of about 700 ms. A small HTTP server
returns a calibrated probability that the request is solvable by the mid-tier model.

## Weights

Weights are not stored in git. Download them from either source:

- Hugging Face (primary): https://huggingface.co/bthe0/switchyard-decider

  ```sh
  hf download bthe0/switchyard-decider --local-dir switchyard-decider
  ```

- GitHub release v1.0: https://github.com/bthe0/switchyard-decider/releases/tag/v1.0

  ```sh
  gh release download v1.0 -R bthe0/switchyard-decider -D switchyard-decider
  ```

Assets: `model.safetensors` (about 792 MB, fp16), `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`calib.npz`.

## Results

Validation set: 1,994 held-out prompts (easy vs hard pairs; medium excluded from AUC).

| model | AUC (labels) | AUC (arena) | AUC (coding) | hard → frontier @40% | easy → mid-tier @40% |
|---|---|---|---|---|---|
| **this model (step 3600)** | **0.886** | **0.769** | **0.929** | **0.693** | **0.862** |
| previous LLM decider (0.8B, MLX 4-bit) | 0.687 | | | 0.538 | 0.712 |

The checkpoint was chosen on this validation set, so these numbers are slightly optimistic. Full tables, the
comparison with about 40 OpenRouter decision models and latency figures are in [docs/benchmarks.md](docs/benchmarks.md).

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
benchmarks/           decision-model benchmark, offline evaluation, latency
tests/                unit tests for the scoring functions (numpy only)
docs/                 architecture, serving, training, benchmarks, data, model card
```

## Documentation

- [Architecture](docs/architecture.md): model, heads, hardness scoring and calibration
- [Serving](docs/serving.md): install, run, HTTP API, running as a service
- [Training](docs/training.md): pipeline order, script inputs and outputs, hardware
- [Benchmarks](docs/benchmarks.md): accuracy, teacher comparison, speed
- [Data](docs/data.md): sources, licenses and labelling
- [Model card](docs/MODEL_CARD.md): the Hugging Face model card

## License

Code and weights are released under the [Apache-2.0](LICENSE) license, matching the ModernBERT base model. The
training labels are outputs of commercial models, and some providers' terms restrict using outputs to build competing
models; read [docs/data.md](docs/data.md#license-and-usage-risk) before any commercial use.
