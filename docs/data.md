# Data

Datasets are not included in this repository.

## Sources

Prompts came from public datasets, used under their licenses:

| dataset | license |
|---|---|
| WildChat-1M | ODC-BY-1.0 |
| LMArena arena-human-preference-140k | CC-BY-4.0 |
| OpenAssistant oasst2 | Apache-2.0 |
| Magicoder-Evol-Instruct-110K | Apache-2.0 |
| CodeFeedback-Filtered-Instruction | Apache-2.0 |
| SWE-bench | MIT |
| SWE-Gym | MIT |
| SWE-rebench | CC-BY-4.0 |

Only the first user turn of each conversation is used. Prompts are filtered to English, 40 to 6,000 characters and
mostly ASCII, and deduplicated by a hash of the lower-cased, whitespace-normalised text. Validation prompts are
excluded from every training source.

## Labelling

Three kinds of labels feed the three heads (see [architecture.md](architecture.md)):

- Arena labels: arena battle outcomes (`winner_model_a` / `winner_model_b` in `judge_battles.parquet`) mapped to
  hard, easy or tie.
- LLM-judge labels: LLM judges (Anthropic Claude Haiku, OpenAI GPT-6-luna decisions) rated prompts easy, medium or
  hard. Several judges are averaged into a soft distribution.
- Gold labels: OpenRouter decision models (`nace-ai/drex-max`, `perplexity/pplx-decider`) gave a probability that a
  mid-tier model can solve the request. Each teacher's scores are converted to percentiles before training.

The three-class `label` used for metrics combines the arena outcome with the primary judge label: hard if either says
hard, easy if both say easy, otherwise medium.

## License and usage risk

Code and weights are released under Apache-2.0, matching the ModernBERT base model. The training labels are outputs of
commercial models. Some providers' terms restrict using outputs to build competing models. This model is a
binary-ish routing classifier, not a general model, but check the relevant terms before any commercial use. The
WildChat (ODC-BY) and CC-BY sources require attribution, which the list above provides.
