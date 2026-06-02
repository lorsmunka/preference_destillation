# Preference Distillation

University thesis proof of concept: distill domain-specific knowledge from Gemma 3 4B into a small transformer using reduced vocabulary. Autoregressive generation is intentional—simpler architectures (BERT, classifiers) could classify, but this proves the technique works for generation.

## Task

Reddit comment classification (tone, sentiment, safety, toxicity) → deterministic JSON output.

## Architecture

- **Teacher**: Gemma 3 4B generates training data with logits
- **Student**: Small transformer with reduced output vocabulary (~525 tokens)
- **Loss**: KL divergence (match teacher distribution) + Cross-entropy (match predicted token), ratio anneals from 0.9→0.1

## Project Structure

Everything is one importable package, `library/`. Researcher entry scripts live at the repo root and import from it (no queue JSON; configs are Python).

```
library/
  shared/      paths/constants (config), vocabulary (reduced-vocab builder), device, metrics/, logging/
  domain/      One self-contained package per domain (reddit / math / post-gen) + registry.
               Each holds its Domain class (teacher prompt, stop token, task metric, vocab
               slice) in __init__.py and a corpus.py producing its {"text": ...} input jsonl.
  model/       Transformer architecture + StudentModel (checkpoint load + generate)
  data_gen/    DistillationDataGenerator (teacher data generation)
  training/    Trainer + TrainingRunner + BatchHandler
  tooling/     Analysis facade + Run / RunStore / Experiment + render (plots, tables)
train.py           training entry — edit the RUNS list
generate_data.py   data-generation entry — edit the JOBS list
analyze.py         analytics/eval entry — Analysis / RunStore
runs/        per-run artifacts: info.json, logs/ (JSONL), checkpoints/
batches/     distillation data (JSONL)
```

## Running

```
python train.py          # run configs are a Python RUNS list (not a JSON queue)
python generate_data.py  # JOBS list
python analyze.py         # or: from library import Analysis, RunStore
```

Press `%` (or Ctrl+C in headless mode) for graceful exit (saves a temp checkpoint).

## Configuration

Run configs are plain Python — one dict per run in `train.py` / `generate_data.py`. Paths and
constants live in `library/shared/config.py` (torch-free). Key per-run params: `epoch_count`,
`batch_size`, `learning_rate`, `kl_ratio_start/end`, `distillation_temperature_start/end`,
`eval_top_k`, `eval_cap_multiple`.

## Evaluation

Single facade: `from library import Analysis`. Accuracy types: **teacher-forced** (ground truth
as context, optimistic), **student** (own predictions, autoregressive), **task accuracy** (was
"classification accuracy") + structural validity rate. Distribution metrics (top-k, teacher /
student entropy, perplexity, mean target rank) and natural-termination / length ratio are
computed **during eval and logged** — post-run analysis only reads logs, never reloads the
model. The `experiment` tag groups seed-replicate runs into cohorts (`Experiment`: mean ± std).

## Code Style

- Full variable names (`index` not `idx`)
- Self-documenting over comments
- Simple, beginner-friendly, no over-engineering
