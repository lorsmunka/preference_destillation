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
  training/    Trainer (the loop) + TrainingRunner + BatchHandler, with the loop's parts beside it:
               schedules (annealed dials), losses (divergence zoo + CE mix), example_encoder
               (example -> tensors), evaluator (eval passes), checkpoints (save / load)
  tooling/     Analysis facade + Run / RunStore / Experiment + render (plots, tables)
train.py           training entry — edit the RUNS list
generate_data.py   data-generation entry — edit the JOBS list
analyze.py         analytics/eval entry — Analysis / RunStore
experiments/ one script per named experiment: its RUNS plus the analysis that reads
             them back (e.g. postgen_kl_anneal.py), with measured tables in results/
runs/        per-run artifacts: info.json, logs/ (JSONL), checkpoints/ — on disk only (gitignored);
             what git keeps is experiments/results/*.csv
batches/     distillation data — batch_N.npz (bf16 logits, bit-exact vs the teacher; legacy .jsonl still readable)
```

## Running

```
python train.py          # run configs are a Python RUNS list (not a JSON queue)
python generate_data.py  # JOBS list
python analyze.py         # or: from library import Analysis, RunStore
python experiments/postgen_kl_anneal.py   # one experiment: trains what is missing, then reports
```

An experiment is a script, not a config file. Copy an existing one in `experiments/`,
edit its `RUNS` (a different architecture is just different `hidden_dim` / `num_layers` /
`num_heads`), and run it — `run_training` skips runs that already completed, so the script
stays re-runnable and doubles as the record of what the experiment was. Scripts there start
with the repo-root `sys.path` line so they run without installing the package.

Graceful exit: the dashboard's Stop button (`python progress.py`, creates a `STOP` file) or Ctrl+C in the run's window — the run saves a resume checkpoint and the script exits; re-running resumes. The reason lands in `queue/last_stop.json` and on the dashboard.

## Configuration

Run configs are typed objects — `TrainingRun(...)` / `GenerationJob(...)` (`library/specs.py`),
one per entry in the `RUNS` / `JOBS` lists. `domain=` takes a **Domain object** (a singleton
like `MATH_WORD_PROBLEM`, a fresh `RedditSentimentDomain()`, or your own subclass — no magic
strings), everything else has a default. Required: `run_name` (training) + `domain`. Key params:
`epoch_count`, `batch_size`, `learning_rate`, `kl_ratio_start/end`,
`distillation_temperature_start/end`, `eval_top_k`, `eval_cap_multiple`. **You pass the Domain
object everywhere** — to specs, and (for a custom domain) to analysis: `Analysis(run).evaluate(domain=MyDomain())`
/ `infer(domain=...)`, `StudentModel.from_run(run, domain=...)`, `build_input_vocabulary(MyDomain())`.
The object is used directly; `info.json` records `domain.name` (string) for paths. Built-in runs
auto-resolve their name (zero ceremony); the name→object registry is an internal detail for that,
not part of the front-door API (`register_domain` exists in `library.domain` as an optional escape
hatch, never required). A plain dict config still works (its `domain` name resolves to a built-in).
Each Domain owns `max_steps` / `max_input_tokens` / `max_seq_length` — no name-keyed config dicts.
Paths/constants live in `library/shared/config.py` (torch-free).

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
