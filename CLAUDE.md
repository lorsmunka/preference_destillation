# Preference Distillation

BSc thesis (Óbuda University, due November 2026): a modular **framework** that distils a large LLM's
domain knowledge into **small transformers trained from scratch** with a task-derived **reduced output
vocabulary**, with fine control over architecture and training schedule and analytics of how the
distillation went. Autoregressive generation is intentional (classifiers could classify; this shows the
technique works for generation).

- **Part 1 — the method across domains** (done, partly being redone): reddit comment → JSON labels, math
  word problems (v1), post-generation, long-context recall (Virág). E-01 compared KL/CE anneal arms.
  **Caution (code review X-C1, 2026-10-08):** the reddit and math students only learned the class prior
  (accuracy at the majority baseline), so their task-accuracy results are floor effects. Diagnosis is the
  current next step.
- **Part 2 — distilling an agent** (in progress): two agents, each = distilled student + harness: the
  **agentic math tutor** and the **coding agent** (added 2026-10-20: find / explain code in a synthetic project).
  Direction (2026-10-20): both work end to end locally, then the large data runs on rented GPU time
  (`thesis_work/cloud_runbook.md`), including a student size sweep (~1M to ~100M parameters,
  `experiments/_common.STUDENT_SIZES`). Status, decisions and next steps: the STATUS block at the top of
  `thesis_work/running.txt`.

## Architecture

- **Teacher**: Gemma 3 4B (part 1: reddit, math v1, post-gen); Gemma 4 12B (Virág, math tutor, coding agent).
  Local runs: RTX 4070, 12 GB, NF4. The large runs of both agents go to rented GPU time (bf16), one teacher
  setup per data file (`library/agent/runner.teacher_setup`). GPU runs only while the user is present.
- **Student**: small transformer from scratch (defaults in `library/shared/config.py`), reduced output
  vocabulary (reddit ~525 tokens; math tutor 18,357 from `math_tutor/output_tokens.json`), compact input
  vocabulary with an unknown-token row.
- **Loss**: forward KL to the teacher's distribution + cross-entropy (at T=1), KL ratio annealed
  (e.g. 0.9 → 0.1), temperature annealed. Other divergences (reverse KL, JS, TVD) available.
- **Teacher data**: generated (teacher writes) or teacher-forced (teacher scores a gold target in one
  pass); bf16 logits over the reduced vocabulary, with the vocabulary's fingerprint and per-record
  metadata (split, item_id, turn_type, teacher mode) stored in each batch.

## The math tutor (part 2) in one paragraph

Items = word problem + executable `solve()` (custom AST interpreter, never exec) + up to 3 follow-ups.
Conversation blocks `<task> <solution> <question> <explanation>`, newest input `<message>`; the model writes
one block and its opening tag is the route: `<solution>` (new task or what-if; pupil-given numbers = a new
task), `<explanation>` (question about the working), `<out_of_scope>` (the **harness** replies with a fixed
sentence). A new task drops the question side thread. The harness checks every solution line / explanation
and re-asks once. Facts-first solutions are worded by Gemma and checked against the code's trace. Follow-up
messages come from a **template bank** (`questions/bank/*.jsonl`, 3,418 templates after the exclusions in
`questions/excluded.jsonl`) filled from the shown solution. **v1 scope is frozen** (deferred ideas: the v2
wish list in `thesis_work/domain_specs.txt`). Splits: test = 10% of families (family = human seed);
held out = any turn with shape add>mul / sub>div / sub>add; test rewordings are an **answer-only
paraphrase test** (no labels or teacher data); SVAMP (735) is the human test. Training data = Gemma's own
replies in simulated conversations (decided 2026-10-13): `slice.py` (the 1,000-problem trial), `full.py` (every item).

## The coding agent (part 2) in one paragraph

20 hand-written base projects (`code_agent/projects/`: 12-15 files of 20-50 lines, ~400-500 lines each, with
`project.json`: ~20-30 topics with their true places, and renames): 16 Python (8 menu + JSON apps, 8 added
2026-10-21 with more classes and algorithms) and 4 plain HTML sites (`html_files.py`: places = elements with an id,
~7,200 tokens each vs ~5,300-6,100 for Python). `project.py` makes seeded randomized versions
(renames, shuffled functions / methods / files / sections of `<main>`) whose true places are found again by parsing. Requests: find or
explain, by meaning (a topic) or by name; 4 per version (`requests.py`). Two steps (`harness.py`): the model reads
the whole project and writes `<scope>` (file + line ranges, checked); the harness narrows the context to it; the
model writes `<explanation>` or `<found/>` (checked: no made-up names). Scored: the scope against the true places,
the explanation against Gemma. Test = 4 whole base projects (shop, todo, spreadsheet, travel_site). No edit actions (user, 2026-10-20).

## Project structure

```
library/
  shared/      config (paths/constants, torch-free), vocabulary, batch_files, metrics/, logging/
  domain/      one package per domain + registry; math_tutor/ = items, renumber, prune, annotations,
               labels (facts-first label pass), conversation (context rules + training records),
               checks, questions (+ questions/bank), threads (explanation pass), harness, agentic (domain),
               slice / full (the conversation passes), demo; code_agent/ = projects, project (+ html_files), requests,
               harness, checks, corpus, vocabulary, demo
  agent/       domain-free parts of an agent: question_templates (template engine), runner (Request,
               TeacherModel, StudentAgentModel, run_sessions, teacher_setup)
  model/       Transformer (KV-cached decoding) + StudentModel (checkpoint load + generate)
  data_gen/    DistillationDataGenerator + ModelHandler (generate / teacher-forced scoring)
  training/    Trainer, TrainingRunner, BatchHandler (split_batch_files), schedules, losses,
               example_encoder, evaluator, checkpoints
  tooling/     Analysis facade + Run / RunStore / Experiment + render + significance
tests/        unittest suite (stdlib; pytest is not installed)
experiments/  one script per experiment (RUNS + analysis), results in experiments/results/*.csv
agents/       briefs and evidence of past agent runs (agents/README.txt); agents/work/ = their scripts
runs/, batches/  on disk only (gitignored)
thesis_work/  thesis docs: running.txt (STATUS), findings.txt, domain_specs.txt, code_review.html,
              literature_review.html (+ literature.bib), thesis_positioning.html, harness_deck.html,
              harness_chat.html (both harnesses replayed step by step: python harness_chat.py),
              thesis.md (the draft: never overwrite), thesis_description.txt (official topic text)
```

## Running

```
python -m unittest discover -s tests      # all tests (CPU, ~20 s)
python train.py                           # RUNS list; returns early if interrupted
python generate_data.py                   # JOBS list
python analyze.py                         # zero args: summary of the first completed run
python experiments/<name>.py              # trains what is missing (completed runs are skipped), then reports
# math tutor data, in this order (labels and threads need the GPU):
python -m library.domain.math_tutor.benchmarks
python -m library.domain.math_tutor.prune
python -m library.domain.math_tutor.renumber
python -m library.domain.math_tutor.labels
python -m library.domain.math_tutor.threads
python -m library.domain.math_tutor.agentic      # output vocabulary (after the threads)
python -m library.domain.math_tutor.harness      # Gemma as tutor on test conversations (GPU)
python -m library.domain.math_tutor.full conversations   # the full pass (cloud), then experiments/math_full.py
python -m library.domain.math_tutor.demo         # talk to a trained tutor student
# coding agent:
python -m library.domain.code_agent.requests     # items.jsonl
python -m library.domain.code_agent.harness      # Gemma's pass -> conversations/v1.jsonl (cloud); "pilot_nf4 16 1" = local trial
python -m library.domain.code_agent.vocabulary   # output vocabulary (after the pass)
python experiments/code_agent_trial.py           # corpus, teacher data, students, report
python -m library.domain.code_agent.demo         # talk to a trained coding-agent student
```

Graceful exit: the dashboard's Stop button (`python progress.py`, creates `STOP`) or Ctrl+C: the run saves a
resume checkpoint and exits; re-running resumes (a mismatched or stale resume point is refused).

## Configuration

Run configs are typed objects — `TrainingRun(...)` / `GenerationJob(...)` (`library/specs.py`), one per
entry in `RUNS` / `JOBS`. `domain=` takes a **Domain object** (no magic strings); required: `run_name`
(training) + `domain`. Key params: `epoch_count`, `batch_size`, `learning_rate`, `kl_ratio_start/end`,
`distillation_temperature_start/end`, `max_training_examples`, `max_test_examples` (fixed test tail,
default 1024), `shuffle_seed` (default None = file order), `eval_top_k`, `eval_cap_multiple`. Each Domain
owns `max_steps` (a runaway guard: longest expected output + ~35%); there is no input-length limit
(rotary positions grow on demand; each run records `longest_training_sequence`). Test set: a fixed tail
of the batch directory, independent of `max_training_examples`; per-record `split` labels win when present.

## Evaluation

Single facade: `from library import Analysis`. Accuracy types: **teacher-forced** (ground truth as
context, optimistic), **student** (own predictions, autoregressive), **task accuracy** + structural
validity. **Always read task accuracy next to the majority baseline and the excess over it** (logged per
eval epoch). Distribution metrics (top-k, teacher/student entropy, perplexity, mean target rank) are
computed during eval and logged; analysis only reads logs. Cohorts = experiment tag + configuration minus
seed, completed runs only, mean ± sample std; pairwise tests with Welch + Bonferroni
(`library/tooling/significance.py`). For the agent: report own-history (autoregressive) accuracy next to
gold-history accuracy; their gap is the cost of compounding errors.

## Code Style

- Full variable names (`index` not `idx`)
- Self-documenting over comments
- Simple, beginner-friendly, no over-engineering
- Never open accumulated data files with "w"; results CSVs go through `experiments/_common.write_rows`
