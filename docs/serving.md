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

- https://github.com/bthe0/switchyard-decider/releases/tag/v1.0

  ```sh
  gh release download v1.0 -R bthe0/switchyard-decider -D switchyard-decider
  ```

The directory must contain `model.safetensors` (about 792 MB, fp16), `config.json`, `tokenizer.json`,
`tokenizer_config.json` and `calib.npz`.

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

Both fields are optional. The model input is the first 3,000 characters of `opening_task` and the full
`latest_follow_up`, joined by a blank line, then truncated to 768 tokens.

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

About 25 ms p50 on Apple-silicon MPS for typical requests. Measured numbers per device are in
[benchmarks.md](benchmarks.md#speed); `benchmarks/bench_speed.py` reproduces them.
