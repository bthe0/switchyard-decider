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
| `TRUNC` | `ht` | `ht` keeps the first and last `MAXLEN/2` tokens of the first 20,000 characters; `head` keeps the first `MAXLEN` tokens (v1.0 and v1.1) |
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

## Released run (v1.0)

- 153k rows, 2 epochs planned, lr 3e-5, max length 768, bf16, batch 8 with 4 gradient-accumulation steps.
- Hardware: a single NVIDIA RTX 5070 Ti (16 GB).
- Stopped early at step about 4,450 of 9,568. Eval AUC peaked at step 3,600 (0.886) and was flat to declining after
  that.
- Weight-averaging ("model soup") of five earlier variants gave +0.009 AUC over the best single run. The gold-label
  head gave +0.027.

## v1.2 retune

v1.2 fixes v1.1's length shortcut: v1.1 read only the first 768 tokens and rated almost any long paste as hard. The
v1.2 scripts are in `training/retune/`. Run them from a data directory with the repo on the path, for example
`PYTHONPATH=/path/to/repo python /path/to/repo/training/retune/build_aug.py`, or after `pip install -e ".[train]"`.

| step | script | inputs | outputs |
|---|---|---|---|
| 1 | `benchmarks/holdout/make_holdout.py` | `cores.py`, `gens.py` | `holdout.jsonl` (frozen, committed) |
| 2 | `holdout_qa.py`, `leak_check.py`, `leak_bow.py` | holdout, training and augmentation files | leakage report (stdout) |
| 3 | `build_aug.py` | `train3.jsonl` | `aug/train4.jsonl`, `aug/aug_train.jsonl`, `aug/devaug.jsonl` |
| 4 | `train4.py` (GPU) | `train4.jsonl`, `val2.jsonl`, `devaug.jsonl`, `aug_train.jsonl`, `et.jsonl` | `$OUT-final/`, `$OUT_dump.npz` |
| 5 | `refit_v12.py <dump> <v1.1 calib.npz> --out-prefix <p>` | dump, `val2.jsonl`, augmentation files | `<p>_aw{None,1,3}.npz`; prints val2 and dev-aug AUC |
| 6 | `embed_set.py <model_dir> <rows.jsonl> <out.npz>` | model, rows | served logits and embeddings |
| 7 | `accept_final.py <calib> <holdout.npz> <v1.1_holdout_h.npy> <fresh.npz> <fresh.jsonl>` | step 5 and 6 outputs | holdout AUC with paired bootstrap, pairs, wrong-side counts |
| 8 | `bench_latency.py <model_dir>` | model, holdout | end-to-end p50/p90 |

Augmentation (`build_aug.py`) makes 4,653 label-preserving rows: easy asks wrapped in long logs, JSON or code pastes;
terse hard questions; and matched pairs that put an easy ask and a hard ask after the same paste. It adds 20k
replay rows sampled from `train3.jsonl`, and holds out a dev-aug split built the same way.

Run: fine-tune from the v1.1 weights for 1 epoch, `BASE=<v1.1 dir> OUT=r10ht LR=1e-5 TRUNC=ht python train4.py`, on
one NVIDIA RTX 5070 Ti. It took about 45 minutes and cost about $0.25 rented. The ridge probe was then refit on the
original probe sample only (`aw=None`).

Selection rule, recorded before any holdout metric was computed: take the highest dev-aug AUC among runs with val2
AUC ≥ 0.8902 (v1.1's 0.8952 minus 0.005). Only the chosen run was scored on the holdout.

| run | LR | input cut | val2 / dev-aug (aw=None) | augmentation-weighted probe (aw=1/3) |
|---|---|---|---|---|
| r5 | 5e-6 | head | 0.8936 / 0.9463 | fails val2 gate |
| r10 | 1e-5 | head | 0.8968 / 0.9483 | fails val2 gate |
| r20 | 2e-5 | head | 0.8987 / 0.9505 | fails val2 gate (0.8851) |
| **r10ht** | 1e-5 | head+tail 384+384 | **0.8966 / 0.9576** | fails val2 gate (0.8832) |

Results are in [benchmarks.md](benchmarks.md#holdout-v12).
