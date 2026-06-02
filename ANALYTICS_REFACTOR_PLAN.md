# Analytics / Eval / Inference Unification — Plan (TLDR)

**Goal:** collapse the 4–6 overlapping tooling systems (thesis §4.2 admits this) into one importable library with a single researcher-facing facade. Replace `input()` menus + queue JSON with Python entry scripts that import classes. Start with **analytics + logging**; data-gen/trainer-as-classes come later but reuse the same core.

**Constraints (from user):** clean-slate OK, breaking changes OK, no back-compat needed, will re-run experiments from scratch. Renames OK (`classification_accuracy → task_accuracy`). Move top-k to eval-time. Keep full configurability but in **Python, not JSON**.

**Scope guard:** inference + demo scripts exist only for **TDK demos, debugging, and testing — this is not a deployment/serving environment** (matches thesis §3.4.1: deploy concerns like unknown-token handling are explicitly out of pipeline scope). So `infer()` stays human-readable (teacher-vs-student outputs + speed/accuracy for the demo); do **not** build serving/API endpoints, throughput batching, or production model-loading paths.

**Core principle — compute once, read-only post-run:** any metric that needs a forward pass is computed **during the eval pass** and written to the run logs — including top-k acc / teacher-student overlap / mean target rank (the eval already holds student *and* teacher logits per step, so they're ~free). **Post-run analysis only reads logs + `info.json`; it never reloads the model.** Re-running the model is reserved for the interactive demo/debug path (`infer()` / `examples()`). This deletes the standalone top-k tool.

---

## Status — built as `library/` (analytics + logging + eval kit + full migration)

> Final layout: the package is **`library/`** (umbrella name temporary). `pdkit` was reorganized into stage subpackages: `library/{shared (config, vocabulary, device, metrics/, logging/), domain, model (transformer + student), data_gen, training, tooling (analysis, run, render)}`. Entry scripts `train.py` / `generate_data.py` / `analyze.py` at repo root. `import library` is torch-free; verified post-restructure (analytics still reproduces the thesis, all chains import, StudentModel stub runs).


Implemented as an additive package `pdkit/` (non-breaking: reads existing `runs/` + JSONL schema; old dirs and the live trainer untouched). `import pdkit` does **not** import torch.

**Verified against the 120 real logs / synthetic data:**
- `pdkit.metrics` — task metrics behind one `TaskMetric`/`MetricResult`; **JSON-validity bug fixed** (invalid output counts as wrong + `validity_rate`); distribution fns (entropy, perplexity, top-k, overlap, rank) and generation fns (termination, length ratio).
- `pdkit.logging` — one typed record schema + reader (maps old `classification_accuracy`→`task_accuracy`; new distribution fields default `None`) + drop-in `RunLogger`. Round-trips.
- `pdkit.run` — `Run` + `RunStore` (one discovery + `resolve_checkpoint`) + `Experiment` cohorts. **Reproduces thesis Table 2 to the decimal** (e.g. sent-8k CE 88.60±0.37 / 64.56±2.29 / 97.68±0.34) using population std.
- `pdkit.domains` — `Domain` registry replaces the scattered string dispatch; optional `structural_validity` (structured domains only).
- `pdkit.analysis.Analysis` facade + `Comparison`; `pdkit.render` tables + plots (PNGs render from real logs).
- Entry script `analyze.py` replaces the `input()` menus.

**Ported but NOT runnable in this env** (no batch data, gated Gemma tokenizer) — compiles + imports, logic ported from the working trainer/top-k/inference:
- `pdkit.student.StudentModel` (one model build + checkpoint load **with the rotary-buffer pop fix** + free-gen rollout) and `Analysis.evaluate()/infer()` (eval-time distribution/entropy/termination/length, 2× cap). Needs a machine with batches + tokenizer to validate end-to-end.

**Whole project migrated onto pdkit (done):**
- `training/trainer.py` + `training_runner.py` now log via `RunLogger`, score via `get_domain().task_metric()`, compute eval-time distribution/entropy/perplexity (free, from the teacher-forced pass) + termination/length (capped free-gen rollout, `eval_cap_multiple`), rename `classification_accuracy`→`task_accuracy`, and plot via `pdkit.render`.
- `distillation_data_generation` dispatches prompt/stop via `pdkit.domains`; `QueueRunner`→`DistillationDataGenerator`.
- **Queue JSON replaced by Python entry scripts** `train.py` / `generate_data.py` (configs in Python). Bare `sys.path`/sibling imports replaced by package-qualified imports.
- **Deleted:** `analysis/`, `experimental_analysis/`, `top-k-accruacy-analsy/`, `shared/logger.py`, `shared/*_accuracy.py`, the queue mains + JSON (git-recoverable).
- Verified: full source compiles; training + data-gen chains import; `StudentModel` path runs on a stub model; analytics still reproduces the thesis.

**Still TODO (needs a real run / later):** smoke a 10-batch `train.py` + a small `generate_data.py` on a machine with batches + the Gemma tokenizer before long runs; optionally fully unify the trainer's eval rollout with `StudentModel` (kept separate to avoid a blind merge of the hottest path); move vocab token data out of `shared.Utilities` into the domains (currently a bridge import).

---

## 1. Current state — the fragmentation

| System | Dir | Entry | Purpose | Overlap |
|---|---|---|---|---|
| Single-run viz | `analysis/` | `input()` menu | log plots, summary, manual eval, inference | reads global `LOGS_DIR` (stale vs `runs/`) |
| Multi-run | `experimental_analysis/` | `input()` menu + CLI | compare table, overlay plots, scaling, HTML report | best run model (`RunData`) lives here |
| Top-k | `top-k-accruacy-analsy/` | argparse + `input()` | distribution metrics (top-k acc, overlap, rank) | re-loads model + batches independently |
| Trainer eval | `training/trainer.py` | (in train loop) | token acc + task acc + confusion | owns the "real" eval logic |
| Data gen | `distillation_data_generation/` | queue JSON | teacher generation | own prompt/stop dispatch |
| Train | `training/` | queue JSON | run training | own model construction |

### Duplication inventory (concrete)

| Logic | Copies | Locations |
|---|---|---|
| `list_runs()` / discover runs | **4** | `analysis/evaluate_model.py`, `analysis/inference.py`, `experimental_analysis/run_data.py` (`load_all_runs`, best), `top-k/main.py` |
| `get_checkpoints()` / resolve checkpoint | **4** | same files; `top-k resolve_checkpoint` is most complete (latest/temp/epoch/path) |
| Build `Transformer` from run config | **4** | `evaluate_model.Evaluator.__init__`, `inference.main`, `top-k.evaluate_run`, `training_runner.run` (source of truth) |
| Load checkpoint (pop `rotary_embedding.*_cached`) | **3 variants** | trainer ✓pop, top-k ✓pop+report, evaluate_model/inference ✗**no pop** (latent bug) |
| Student autoregressive rollout | **3** | `trainer._eval_single_example`, `inference.generate_student`, `evaluate_model.Evaluator.evaluate` |
| Teacher-forced single-pass logits slice (`logits[0, L-1:L-1+steps]`) | **3** | `trainer.train_single_example`, `trainer._eval_single_example`, `top-k.evaluate_example` |
| `_prepare_step_data` (token_id, logit vector, target idx) | **2** | trainer + top-k (inline) |
| JSONL load + parse | **2 parsers** | `analysis` (dict + filter by `type`), `run_data` (typed dataclasses) |
| Prompt builder dispatch | **3** | `inference.create_teacher_prompt`, `ModelHandler.build_prompt`, `Utilities.create_*_prompt` |
| Stop-token dispatch | **2** | `inference.DOMAIN_STOP_TOKEN`, `ModelHandler.is_stop` |
| Domain limits (`MAX_GENERATION_STEPS`, `MAX_INPUT_TOKENS`, `MAX_SEQ_LENGTH`) | scattered dicts | `config.py` + local dict in `inference.py` |
| Task-metric selection (if/elif domain) | **1** but coupled | `trainer._evaluate_batches` |

### Structural problems
- **`input()`-driven + queue-JSON-driven** everywhere → not scriptable, not testable, not importable.
- **`sys.path.insert` hacks** in ~7 files; `training/` modules import by bare name (`from model import …`) — only works via path insert + `os.chdir`. `pyproject.toml` exists but package is **not installed**.
- **Global config state:** `LOGS_DIR="./logs12b"` (doesn't match `runs/{name}/logs/`), `INFERENCE_TEMPERATURE=0` (can't vary per call), vestigial arch defaults "overridden by queue".
- **Domain = magic string** dispatched in ≥8 sites; adding a domain means editing many files.
- **`Utilities`** is a god-class (vocab + prompts + example responses + logit extraction + token sections).
- **Metric calculators** share an implicit interface but no base type; `get_confusion_matrices()` returns 3 incompatible shapes; `compute_f1_per_category` guards with `isinstance` to survive it.
- **Compute and render are fused** (analyzers call `matplotlib`/`print`/`input` directly) → numbers can't be reused without a TTY.
- **Wasteful post-run model re-runs:** the top-k tool reloads the model + re-runs batches to compute metrics the eval pass could have produced for free and logged.
- **Experiment cohorts aren't aggregated:** the `experiment` tag (cohort of seed-replicates) is shown only as a column; mean±std / significance live stranded in `thesis_work/build_grafikonok.py` (see §4).

---

## 2. Target architecture

One installed package. **Compute returns typed data; rendering (plot/print/html) is a thin separate layer.** Everything below is reused by the future trainer/generator refactor.

```
preference_distillation/                 # pip install -e . ; import as pd
  domains/          Domain abstraction — kills string dispatch
    base.py         class Domain(ABC): name; prompt(text); stop_token; max_steps;
                    max_input_tokens; max_seq_length; example_responses;
                    prompt_tokens; aux_tokens; task_metric() -> TaskMetric;
                    structural_validity(output) -> bool | None  (structured domains only; free-form -> None)
    reddit_sentiment.py / math_word_problem.py / post_generation.py
    registry.py     register(Domain); get_domain(name) -> Domain
  vocab.py          build_vocabulary(tokenizer, domain, aux_pct)  (from Utilities, domain-driven)
  metrics/
    base.py         class TaskMetric(Protocol): update(pred_tokens, gt); result()->MetricResult
    classification.py / math.py / post_generation.py   (moved from shared/)
    distribution.py topk_acc, overlap, mean_target_rank, entropy(teacher&student),
                    perplexity  — all computed IN the eval pass, no model reload
    token.py        teacher_forced_acc, student_acc (extracted from trainer)
    generation.py   termination_rate, length_ratio  (need the free-gen eval rollout)
  logging/
    records.py      @dataclass TrainBatch|TrainEpoch|EvalEpoch|MiniEval|GenBatch  (SINGLE schema)
                    EvalEpoch/MiniEval gain topk_accuracy/overlap/mean_target_rank fields
    writer.py       RunLogger  (was shared/logger.py) — writes typed records
    reader.py       read_records(run_dir) -> typed lists  (was run_data.parse_*)
  run.py            class Run (was RunData: info + typed logs + properties)
                    class RunStore: all()/completed()/filter()/get(name); resolve_checkpoint()
  student.py        class StudentModel: from_run(run, ckpt); .forward_logits();
                    .generate(text); .teacher_forced(example); .step_metrics(example, topk)
  config.py         PATHS ONLY (no LOGS_DIR / INFERENCE_TEMPERATURE globals)
  analysis.py       FACADE (below)
  render/           plots.py, tables.py, report_html.py  (consume results, no compute)
```

### The single facade

```python
from preference_distillation import Analysis, RunStore

a = Analysis("tsc-scale-128h-4L-36M")        # accepts run name | Run | checkpoint path
a.summary()                                  # -> RunSummary (typed, .print())
a.training_curves()                          # -> arrays; a.plot_training() renders
a.evaluate(split="test",
           metrics=["token","task","topk"])  # -> EvalResult  (top-k now eval-time)
a.examples(batch=300, n=10)                  # -> [ExampleResult] (student/TF/GT, colored)
a.infer(["F**k you"], compare_teacher=True)  # -> InferenceResult (speed + outputs)

Analysis.compare(RunStore.completed())\      # multi-run, same class entry
        .table().csv("out.csv").html().scaling()
```

- `Analysis` is the **one entry point** the user asked for; single-run via constructor, multi-run via `Analysis.compare(...)` → internal `Comparison`.
- Every method **returns a result object**; `.print()/.plot()/.html()` live on results or `render/`. Numbers are testable without matplotlib or a TTY.
- All four old tools become thin methods over `Run` + `StudentModel` + `metrics`.

### Domain abstraction (the biggest simplifier)

Replaces `DOMAIN_STOP_TOKEN`, `DOMAIN_MAX_*`, `create_*_prompt`, `is_stop`, `build_prompt`, the vocab if/elif, and trainer's metric if/elif — **one class per domain, registered once.**

```python
class Domain(ABC):
    name: str
    def prompt(self, text: str) -> str: ...
    @property
    def stop_token(self) -> str: ...
    @property
    def max_steps(self) -> int: ...
    def task_metric(self) -> TaskMetric: ...
    # + example_responses / prompt_tokens / aux_tokens for vocab build
```
Adding a domain = new file + `register()`. Enables the thesis's "cleaner domain definitions" and custom domains.

---

## 3. Logging: one schema, two directions

- `logging/records.py` is the **single source of truth** for record shapes; `writer.py` (train side) and `reader.py` (analysis side) both import it. Kills the dict-vs-dataclass double parser.
- `logs_dir` is **always** `runs/{name}/logs/` (a `Run` owns it). Retire global `LOGS_DIR` and the orphaned `logs12b` path; the interactive `LogVisualizer` becomes `Analysis(run).plot_training()`.
- Keep current JSONL-by-type layout (it works); just type it. `RunLogger` keeps the resume/state-file behavior.
- **Extend `EvalEpoch`/`MiniEval` with the distribution metrics** (topk acc / overlap / mean target rank) so they're captured at eval time and never recomputed. `info.json` already carries the `experiment` tag — keep writing it (it's the cohort key, §4).

---

## 4. Experiments (cohorts) — first-class

The `experiment` tag already lives in every `info.json` (set from the run config). It groups **seed-replicate runs**: `purekl-t1`, `purece-t1`, `kl99to50-t1`, `math-8k-*`, `8k-*` = **10 runs each**; postgen sets = 3. This cohort — not the single run — is the real unit of research output: thesis Table 2 and Figs 6–9 / 17 are **per-experiment mean ± std** (plus Cohen's d / Welch t), not per-run numbers.

**Gap:** nothing in `analysis/` or `experimental_analysis/` aggregates by experiment — it's shown only as a column. The cohort math lives stranded in `thesis_work/build_grafikonok.py` (`mean`/`std`, pooled-d, Welch, seed-mean training curves, and a post-hoc top-k recompute). Promote it into core:

- `RunStore.by_experiment() -> {tag: [Run]}`; `Experiment(tag, runs)` with `.stats(metric) -> (mean, std, n)`, `.seed_mean_curve(metric)`, `.compare(other) -> {cohens_d, welch_t, p}`.
- Facade: `Analysis.experiment("purekl-t1").summary()` (mean±std table) and `Analysis.compare_experiments(["purece-t1","purekl-t1","kl99to50-t1"])` → reproduces the thesis comparison **straight from logs** (no model reload, per §Core principle).
- `build_grafikonok.py` becomes a thin renderer over `Experiment`, or is retired.

**Tests: deferred** (user — not needed now). Still keep compute/render decoupled so a `tests/` suite (pure metric/parse/vocab/`resolve_checkpoint` functions on a tiny fixture run) is trivial to add later. Not a blocker for any phase.

---

## 5. Metrics to add (decided this round)

All computed **at eval time and logged** (per Core principle) unless noted; new eval-record fields flow `records.py` → `Run` → facade/cohort automatically.

| Metric | Tier | What / why | Cost |
|---|---|---|---|
| **Structural validity rate** | necessary | Per-`Domain` **optional** (`structural_validity`): only structured domains (sentiment JSON, math scaffold) implement it; free-form → `None`. **Also fixes a bug:** `ClassificationAccuracyCalculator` today drops unparseable student JSON from the denominator → task accuracy is overstated (and inconsistent with math, which counts failures as wrong). | free at eval |
| **Teacher & student entropy** | necessary | Quantifies the "sharp vs soft distribution" claim the whole domain comparison rests on (§3.4.3) — currently unmeasured. | teacher: offline from saved logits; student: free at eval |
| **Perplexity** (student; teacher ref) | nice (free) | Standard LM unit `exp(CE)`, distribution-aware unlike accuracy. Honest note: monotonic transform of CE you already log — comparability with literature, not new signal. | free at eval |
| **Natural termination rate + length ratio** | necessary | Eval forces student to teacher-length, so self-termination is never tested. Free-gen rollout **capped at 2× teacher length** (configurable); no stop token by cap → failure. Same rollout yields student/teacher **length ratio**. | one free-gen rollout: ≈+10% steady-state, bounded +100% on non-terminating examples |
| **Compression ratio** | nice | teacher_params / student_params (34×–860× headline) — currently prose-only. | free (param counts), derived `Run` property |
| **Inference speedup** | nice | teacher TPS / student TPS — `infer()` computes then discards it. | one-time benchmark; loads teacher (demo/benchmark only, never post-run) |

**Enabling change:** add a **free-generation eval mode** (student runs to its own stop token or the 2× cap). This one addition unlocks termination rate + length ratio now, and cheaply unlocks repetition/distinct-n later.

**Parked for your call** (`checklist.md`): reverse-KL / JSD (free, engages MiniLLM [6]), repetition + distinct-n (post-gen; ~free once free-gen rollout exists), gradient norm (train hook), train/eval loss-gap curve (free, already logged), semantic similarity for post-gen (heavier; addresses the thesis §6 limitation). **OOV rate: dropped.**

---

## 6. Migration phases (analytics first)

1. **Package + paths.** Make `pip install -e .` work; delete `sys.path` hacks; rename `training/` bare imports to package-qualified. *(enabler, no behavior change)*
2. **`logging/records.py` + reader/writer** over existing JSONL. Point `RunData`→new `reader`.
3. **`Run` + `RunStore`** = promote `experimental_analysis/run_data.py`; add `resolve_checkpoint` (from top-k). Delete the 4 `list_runs/get_checkpoints` copies.
4. **`StudentModel`** = one model-load + generate + teacher-forced + step-metrics. Delete 3 rollout copies + 4 construction copies; fix the no-pop checkpoint bug.
5. **`domains/` + `metrics/`** = move calculators behind `TaskMetric`; build `Domain` registry (incl. optional `structural_validity`); rewire trainer eval, data-gen, vocab to it. **Compute distribution + entropy + perplexity + validity in the eval pass and log them; add the free-gen eval rollout (termination/length); delete the standalone top-k tool.**
6. **`Analysis` facade + `render/` + `Experiment`** = re-express `analysis/` + `experimental_analysis/` + top-k as methods; add cohort aggregation (`RunStore.by_experiment`, `compare_experiments`) promoted from `build_grafikonok.py`. Old dirs deleted.
7. **Entry scripts** = `main_analysis.py` (or notebook) importing `Analysis`; later `main_train.py` importing `Trainer`, `main_generate.py` importing `DistillationDataGenerator`. Retire queue JSON.

Gate per phase: one real run's plots reproduce **and** a cohort's mean±std matches the thesis tables. (Tests deferred — §4.)

---

## 7. Low-hanging fruit / industry-standard flags

- **Install the package** (pyproject exists, unused) → removes every `sys.path.insert`/`os.chdir`. Highest ROI.
- **Latent bug:** `analysis/evaluate_model.load()` and `inference.py` student load don't pop `rotary_embedding.*_cached` before `load_state_dict` (trainer + top-k do). Centralize in `StudentModel.load`.
- **`LOGS_DIR="./logs12b"`** is stale vs `runs/` layout → the interactive single-run visualizer is effectively orphaned. Remove global.
- **`INFERENCE_TEMPERATURE`/sampling** should be a call arg, not a global constant.
- **`Utilities` god-class** → split into `vocab.py` + per-domain data on `Domain`.
- **`get_confusion_matrices()`** misnamed for math/postgen (returns counts) → unify under `MetricResult` with optional `confusion` field; drop the `isinstance` guard in F1.
- **Delete the standalone top-k tool** — fold its 3 metrics into the eval pass (free there) and log them; post-run never reloads the model just for numbers.
- **Declarative metric columns** (`COLUMN_DEFINITIONS`/`WINNER_METRICS` in `evaluate_runs.py`) are good — keep, move to `render/tables.py`.
- Consider `rich`/`tabulate` for tables, `scipy` for scaling fit (already noted in old plan), and dropping `pynput` exit-listener for a simple `signal`/KeyboardInterrupt handler.

## 8. Naming (user-approved)
- `classification_accuracy` → `task_accuracy` (blast radius: logger schema, `Run`, eval columns, plots, HTML, thesis text — do as one sweep in Phase 5).
- top-k "tool" → `metrics=["topk"]` computed during the eval pass (no separate model load).
- `Evaluator`/`LogVisualizer`/`GenerationAnalyzer`/`TrainingAnalyzer` → methods on `Analysis`.

## 9. Open decisions
- **One class vs two:** `Analysis` + internal `Comparison` (recommended) vs forcing literally one class for single+multi run.
- **Package name:** reuse repo name `preference_distillation` vs short alias `pd`.
- **Render deps:** keep matplotlib-only vs add `rich`/`scipy`.
- **Domain config:** pure-Python `Domain` subclass (recommended) vs dataclass + functions.
- **Eval-time metric set:** which distribution metrics + default `k` to compute & log (recommend top-20 to match thesis). Changing `k` afterward is the one case needing a re-eval — acceptable because it's rare and explicit.
