# Architecture

## Purpose

A local LLM router sends each coding-assistant request either to a cheaper mid-tier model or to a frontier model.
The decision was previously made by a small LLM (0.8B, MLX 4-bit), which took about 700 ms per request and ranked
requests poorly (AUC 0.687). switchyard-decider replaces it with an encoder classifier that answers in about 25 ms
(p50, Apple-silicon MPS) and ranks requests much better (AUC 0.895 with the v1.1 ridge probe).

## Model

The model is `answerdotai/ModernBERT-large` with a sequence-classification head of seven output logits, fine-tuned
with three targets:

| logits | target | loss |
|---|---|---|
| 0–2 | arena-derived easy / medium / hard label | cross-entropy |
| 3–5 | LLM-judge soft label (easy / medium / hard) | soft cross-entropy |
| 6 | "gold" hardness from OpenRouter decision models, percentile-calibrated per teacher | BCE |

The gold logit is trained against the teacher's probability that the request is mid-tier-solvable, so a **higher
gold logit means easier**. Everything downstream uses its negation as a hardness signal. The logit order and this
sign convention are part of the released weights and must not change.

The three heads see different, partially overlapping label sources (see [data.md](data.md)). Training them jointly
lets each source regularise the others; the gold head alone added +0.027 AUC.

## Hardness score

For one request with logits `lg`:

1. `pr = (softmax(lg[0:3]) + softmax(lg[3:6])) / 2`, the mean of the arena and judge distributions.
2. Two percentiles against the validation set: the rank of `pr[2]` (P(hard)) in the sorted validation values `aj`,
   and the rank of `-lg[6]` in the sorted validation values `g`.
3. `hardness = (rank_aj + rank_g) / 2 / n`, a number in [0, 1].

Averaging percentiles rather than raw scores puts the two signals on the same scale without fitting any weights.

## Calibration

A router usually has fixed probability cutoffs tuned for its previous decider. To keep those cutoffs routing the same
share of traffic, the hardness score is mapped onto a target output distribution:

1. `q = rank of hardness in the sorted validation hardness values h / n`.
2. `p_solve = quantile(out, 1 - q)`, where `out` holds the sorted target scores.

If no target is given, `out` is `linspace(0, 1, n)`, so `p_solve` is simply one minus the hardness percentile. With a
target (for example the previous decider's raw scores on the same validation set), the output distribution matches
the old one while the ranking comes from the new model.

All reference arrays live in `calib.npz` next to the weights, with keys `aj`, `g`, `h` and `out`.
`switchyard_decider/calib.py` builds it from validation logits.

## Code

The scoring math lives in pure numpy functions in `switchyard_decider/serve.py` (`request_text`, `hardness`,
`p_solve`), which are unit-tested without torch. Model loading and the HTTP server are in `load_scorer` and `main`,
which import torch and transformers lazily.
