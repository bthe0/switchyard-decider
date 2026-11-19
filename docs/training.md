# Training

Datasets are not in git. All scripts read and write files in the current working directory, so run them from a data
directory, for example `python /path/to/repo/training/train3.py`. Install the extra dependencies with
`pip install -e ".[train]"`.

## Pipeline

| step | script | inputs | outputs |
|---|---|---|---|
| 1 | `training/make_splits2.py` | `decider_train.jsonl`, `or_labels/<model>/*.jsonl`, `val.jsonl`, `judge_battles.parquet` | `train2.jsonl`, `val2.jsonl` |
| 2 | `training/gold_build.py` | `val.jsonl`, `train2.jsonl`, public datasets under `ext_data/` | `gold_pool.jsonl` |
| 3 | `training/gold_label.py <rows.jsonl> <model>` | rows with `id` and `text` | `gold_labels/<model>.jsonl` |
| 4 | `training/build_train3.py` | `gold_labels/*.jsonl`, `train2.jsonl`, `gold_pool.jsonl` | `train3.jsonl` |
| 5 | `training/train3.py` | `train3.jsonl`, `val2.jsonl` | `$OUT/checkpoint-*`, `$OUT/val_logits.npy`, `$OUT-final/` |
| 6 | `training/soup.py <run_dir>...` | trained model directories, `val.jsonl` | `soup/` |
| 7 | `switchyard_decider/calib.py <model_dir> <val_logits.npy> [target_p.npy]` | validation logits | `<model_dir>/calib.npz` |

### 1. Splits and judge labels

`make_splits2.py` joins arena battles (`judge_battles.parquet`) with LLM-judge labels. Each training row gets the
arena label (`arena_y`: easy, tie, hard), a soft judge distribution (`jsoft`) averaged over the annotators listed in
`ANN` (default `luna,haiku,qwen,nemotron`), and `has_j`. Prompts in `val.jsonl` are excluded by normalised-text hash.

### 2. Gold pool

`gold_build.py` samples English prompts of 40 to 6,000 characters from the public datasets in [data.md](data.md),
deduplicated against the training and validation sets, with a quota for code-like prompts.

### 3. Teacher labelling

`gold_label.py` asks an OpenRouter decision model for the probability that each row is mid-tier-solvable, using the
client in `benchmarks/bench_dec.py`. It resumes from existing output.

Environment: `OPENROUTER_API_KEY` (required), `CAP` (stop when account usage reaches this many dollars, default
unlimited), `TRUNC` (prompt characters sent, default 12000).

### 4. Merge

`build_train3.py` converts each teacher's scores to percentiles (each teacher labels a random half of one pool, so
percentiles share a scale) and attaches them as `gp` and `has_g`. Training rows are matched by id
`rl:<first 16 hex chars of sha1(normalised text)>`; gold-pool rows are added with only the gold target.

### 5. Training

`train3.py` fine-tunes the base model with seven logits and the combined loss described in
[architecture.md](architecture.md). The best checkpoint is selected by validation AUC.

| variable | default | meaning |
|---|---|---|
| `BASE` | `answerdotai/ModernBERT-large` | base model |
| `TRAIN` | `train3.jsonl` | training file |
| `LR` | `3e-5` | learning rate |
| `EP` | `3` | epochs |
| `MAXLEN` | `768` | max tokens |
| `LS` | `0` | label smoothing for the arena head |
| `SEED` | `42` | random seed |
| `JW` | `2` | judge-head loss weight |
| `GW` | `2` | gold-head loss weight |
| `ES` | `600` | eval and save interval in steps |
| `OUT` | `ckpt` | output directory |
| `SMOKE` | unset | if set, train 4 steps on 64 rows |

### 6. Weight soup

`soup.py` scores each run with `switchyard_decider/predict.py`, then greedily averages weights, keeping a run only if
the soup's validation AUC improves.

### 7. Calibration

`calib.py` writes `calib.npz`. Pass the previous decider's validation scores as `target_p.npy` to keep existing
router cutoffs; see [architecture.md](architecture.md#calibration).

## Released run

- 153k rows, 2 epochs planned, lr 3e-5, max length 768, bf16, batch 8 with 4 gradient-accumulation steps.
- Hardware: a single NVIDIA RTX 5070 Ti (16 GB).
- Stopped early at step about 4,450 of 9,568. Eval AUC peaked at step 3,600 (0.886) and was flat to declining after
  that.
- Weight-averaging ("model soup") of five earlier variants gave +0.009 AUC over the best single run. The gold-label
  head gave +0.027.
