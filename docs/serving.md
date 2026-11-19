# Serving

## Install

Python 3.10 or newer.

```sh
git clone https://github.com/bthe0/switchyard-decider
cd switchyard-decider
pip install -e .
```

## Weights

Download the weights from Hugging Face (primary) or the GitHub release:

- https://huggingface.co/bthe0/switchyard-decider

  ```sh
  hf download bthe0/switchyard-decider --local-dir switchyard-decider
  ```

- https://github.com/bthe0/switchyard-decider/releases/tag/v1.2

  ```sh
  gh release download v1.2 -R bthe0/switchyard-decider -D switchyard-decider
  ```

The directory must contain `model.safetensors` (about 792 MB, fp16), `config.json`, `tokenizer.json`,
`tokenizer_config.json` and `calib.npz`. The v1.2 weights need the v1.2 package (1.2.0), which applies the head+tail cut
described below. An older server keeps only the first 768 tokens and gives wrong scores for long inputs.

## Run

```sh
switchyard-decider-serve switchyard-decider --port 4126 --host 127.0.0.1
# or
python -m switchyard_decider.serve switchyard-decider
```

The server picks MPS, then CUDA, then CPU. It runs fp16 on GPU and fp32 on CPU, scores one warmup request, then prints
`serving <model_dir> on <host>:<port> (<device>)`. Requests are serialised with a lock, one forward pass at a time.

## API

`POST /v1/systemone`

Request body:

```json
{"state": {"opening_task": "first user message of the task", "latest_follow_up": "most recent user message"}}
```

Both fields are optional. The model input is the first 20,000 characters of `opening_task` and the full
`latest_follow_up`, joined by a blank line and cut to 20,000 characters. If the result is longer than 768 tokens, the
first 384 and last 384 tokens are kept (`head_tail` in `serve.py`), so an ask after a long paste still reaches the
model.

Response body:

```json
{"answers": {"solve": {"noul": 0.84}}}
```

`noul` is the calibrated probability that the request is solvable by the mid-tier model. Lower means harder; a router
sends requests below its cutoff to the frontier model.

```sh
curl -s -X POST http://127.0.0.1:4126/v1/systemone \
  -H 'content-type: application/json' \
  -d '{"state": {"opening_task": "Rename the variable foo to bar in utils.py", "latest_follow_up": ""}}'
```

## Running as a service

On macOS, a launchd agent in `~/Library/LaunchAgents/` with `ProgramArguments` set to the absolute paths of the
`switchyard-decider-serve` executable and the model directory, plus `RunAtLoad` and `KeepAlive` set to true, keeps the
server running across logins.

On Linux, a systemd unit works the same way:

```ini
[Unit]
Description=switchyard-decider

[Service]
ExecStart=/path/to/venv/bin/switchyard-decider-serve /path/to/switchyard-decider --port 4126
Restart=always

[Install]
WantedBy=default.target
```

The server binds to 127.0.0.1 by default and has no authentication; keep it on localhost or behind a proxy.

## Latency

v1.2 end to end on Apple M4 Max MPS (fp16, batch 1, long holdout texts, tokenization included): 62.8 ms p50,
64.9 ms p90. v1.1 measured 64.0 ms on the same harness. Short prompts take about 25 ms p50 through the server (a v1.1
measurement). See [benchmarks.md](benchmarks.md#latency-v12-harness). `training/retune/bench_latency.py` and
`benchmarks/bench_speed.py` reproduce these numbers.
