# Internal evidence fact sheet for thesis chapter 3 (3.1-3.3)

Prepared 2026-11-03 from the repo `D:\preference_destillation` (read only). Every number has a source:
a file:line, a CSV row, a run's `info.json` / `logs/training.jsonl`, or a computation shown here.
"UNVERIFIED" = could not be confirmed from the repo. Confidence: HIGH (read directly from logs/CSV/code),
MEDIUM (from STATUS/ledger notes or estimates), LOW (indirect or single note).

Scratch scripts used (not in the repo): `scratchpad/work/runs_summary.py` (every run's info.json + final
eval), `scratchpad/work/teacher_params.py` (teacher parameter counts from the safetensors headers in the
local Hugging Face cache), `scratchpad/work/cpu_timing.py` (CPU-only inference timing, section 4).
The unittest suite was run on the CPU from a scratch folder: **157 tests, OK, 17 s** (2026-11-03).

---

## 1. Sizes and compression ratios actually achieved

### 1.1 Teachers (HIGH: counted from the safetensors headers of the cached checkpoints)

| Teacher | Total parameters | Language-model part | Source |
|---|---:|---:|---|
| Gemma 3 4B it | 4,300,079,472 (4.30B) | 3,880,263,168 (vision tower 416.9M) | `~/.cache/huggingface/hub/models--google--gemma-3-4b-it/.../model.safetensors.index.json` (total_size 8,600,158,944 bytes = bf16) |
| Gemma 4 12B it | 11,959,730,224 (11.96B) | 11,907,350,320 | `models--google--gemma-4-12B-it/snapshots/707f0a3b.../model.safetensors` (23.9 GB) |

Gemma 4 12B config: vocab 262,144, hidden 3,840, 48 layers, tied embeddings (`config.json`).
Gemma 3 4B input embedding: 262,208 x 2,560 = 671,252,480 (header; matches old thesis.md:192).

### 1.2 Every student trained, by domain (HIGH: `runs/*/info.json` model_info + final `eval_epoch`)

Task accuracy is the final pass; baseline = always giving the most common answer (logged since the
2026-10-08 fix; older runs have no logged baseline, it comes from findings/CSVs).

**Math tutor agent, Gemma 4 12B, full data (the only agent with working students)**

| Run | Params | Data | Task acc. final | Baseline | Excess | Source |
|---|---:|---|---:|---:|---:|---|
| math-full-kd-xs-s1 (64x2) | 2,669,678 | 64,554 train records | .050 | .041 | +.008 (= ~3 of 362 answers: noise) | results/math_full.csv row 2 |
| math-full-kd-s1 (128x4) | 6,106,798 | 64,554 | .340 (best .351) | .041 | +.298 | math_full.csv row 1 |
| math-full-ce-s1 (128x4) | 6,106,798 | 64,554 | .304 | .041 | +.262 | math_full.csv row 3 |
| math-full-kd-m-s1 (256x6) | 16,389,422 | 64,554 | .495 | .041 | +.453 | math_full.csv row 4 |
| math-full-kd-l-s1 (512x8) | 53,732,910 | 64,554 | .580 | .041 | +.539 | math_full.csv row 5 |
| math-agent-kl-m-s1/2/3 | 16,456,494 | 195,980 train / 8,415 test replies (3,147 scored answers) | .700 / .719 / .697 | .042 | +.66 to +.68 | results/math_agent.csv pass 10 |
| math-agent-ce-m-s1/2/3 | 16,456,494 | same | .684 / .688 / .677 | .042 | +.64 | math_agent.csv pass 10 |
| math-agent-kl-to-ce-m-s1/s2 | 16,456,494 | same | in progress (0 passes logged) | | | runs/*/info.json status in_progress (started 2026-11-03) |
| math-full-pilot_nf4-* (xs..xl) | 2.59M-103.07M | ~100 records | 0 | .125 | negative | runs/math-full-pilot_nf4-*/ (pilot, nothing learned by design) |
| slice-* (5.9M-6.0M) | 5,884,756-5,976,428 | 576-15,301 records | 0-3.7% | 6.1-7.4% | all negative | slice_trial.csv, slice_curve.csv, slice_mix.csv |

First loop's 362 test answers come from the confusion matrix of math-full-kd-l-s1 (sum = 362); the CE vs KL
runs score 3,147 answers (math-agent-kl-m-s1 final confusion matrix).

**Coding agent, Gemma 4 12B: only pilots (36-48 training records), nothing learned**

| Runs | Params | Task acc. | Baseline | Source |
|---|---|---:|---:|---|
| code-agent-pilot_nf4-* (xs..xl) | 617,001 - 83,481,513 | 0 | .125 | runs/*/info.json; results/code_agent_trial_pilot_nf4.csv (scopes 0/8 for both students) |
| code-agent-pilot2_nf4-* (xs..xl) | 1,156,680 - 88,840,584 | 0 | .167 | results/code_agent_trial_pilot2_nf4.csv (scopes 0/6 for every size) |

No run on the full v1 data exists (no `runs/code-agent-v1*`, no `batches/code_agent/.../input_vocabulary.json`). HIGH.

**Reddit sentiment, Gemma 3 4B**

| Runs | Params | Data | Task acc. | Baseline | Source |
|---|---:|---|---|---|---|
| E-01 (50 runs, 48x3) | 3,416,397 (96% input embedding, findings.txt F-68a) | 2,048 examples | best .621-.630, final .598-.620 | .626 | findings.txt:296-299 (X-C1); excess best +.003..-.005, final -.006..-.028 (findings.txt:324-327) |
| April 2026 scaling `tsc-scale-*` (7 sizes, full 262,144 input vocab) | 5,266,805 - 129,186,573 | 489,984 train comments, 3 epochs | 86.0% - 87.4% per-field value accuracy | not logged (see note) | git cd807da runs/tsc-scale-*/; thesis_work/grafikonok/data/03_sentiment_scaling_curve.csv |
| April `tsc-reduced-input-proj-5M-64h-8L` | 4,932,493 (input vocab 68,328) | 489,984 | 86.3% ("classification_accuracy" .8633) | ~64% (mean of the four per-field majority shares, computed from its confusion matrices) | git cd807da:runs/tsc-reduced-input-proj-5M-64h-8L/; grafikonok/data/08b_reddit_confusion_matrices_reduced_input.csv |

Note on the April 86%: the April metric (`git show cd807da:shared/classification_accuracy.py`, `update()`
returns 0,0 for a reply that does not parse) skips unparseable replies, and the confusion matrices only hold
8,210 of the 10,016 test comments. Per field on those 8,210: tone 87.4% (majority 66.4%), sentiment 77.7%
(51.0%), safety 90.1% (72.5%), toxicity 90.1% (65.1%) -> mean 86.3% vs 63.8%. If the 1,806 missing replies
are counted wrong, the value accuracy is about 0.863 x 8,210 / 10,016 = ~71% (still above 64%, by ~7 points
rather than ~22). These runs predate the 2026-10-08 code-review fixes and have one seed each. MEDIUM.

So X-C1 (the students only learned the class prior) is true for E-01 (2,048 examples) and the old 8k-example
runs (old thesis.md:165-170: 64.6% classification accuracy at 8,000 examples), not for the April 500k runs.

**Math v1 (templated word problems), Gemma 3 4B**: E-01, 50 runs, 12,719,712 params of which 12,582,912
(98.9%) is a *full* 262,144-token input embedding (runs/e01-math-*/info.json; old thesis.md:219 confirms the
math runs used the full input vocabulary), output 528. At the prior: always "True" .209; numeric answers
5/284 (findings.txt:299-300); best-epoch excess +.021..+.037, 0/10 pairs significant (findings.txt:326-327).

**Post-generation, Gemma 3 4B**: E-01, 15 runs, 14,000,131 params (input 262,144 full, output 26,659).
"Task accuracy" = structural validity (<end> present) .975-.999 (e01_anneal_direction.csv); no majority
baseline exists for free text; own-history token accuracy only .043-.047 (runs/e01-postgen-*). The result
is distribution fidelity (section 6.2), not generation quality.

**Long-context recall (Virág), Gemma 4 12B**: teacher data only (4,968 examples, F-V1), no student. Not in the thesis.

### 1.3 Teacher/student ratios (computed: total teacher params / student params)

| Student | Params | Teacher | Ratio | Works? |
|---|---:|---|---:|---|
| code pilot xs | 617,001 | Gemma 4 12B | 19,384x | no (36-48 records, task 0) |
| math full xs | 2,669,678 | Gemma 4 12B | 4,480x | no (at baseline) |
| math full s (kd / ce) | 6,106,798 | Gemma 4 12B | **1,958x** | **yes: .340 vs .041 (smallest clearly working agent student)** |
| math m (full / agent) | 16.39M / 16.46M | Gemma 4 12B | 730x / 727x | yes: .495; .705 (3-seed KL mean) |
| math full l | 53,732,910 | Gemma 4 12B | **223x** | yes: .580 (largest trained on the full data) |
| math pilot xl | 103,073,772 | Gemma 4 12B | 116x | no (100 records) |
| reddit E-01 | 3,416,397 | Gemma 3 4B | 1,259x | no (at prior) |
| reddit April reduced-input | 4,932,493 | Gemma 3 4B | 872x | yes, on parseable replies (86% vs 64%) |
| reddit April scale smallest / largest | 5.27M / 129.19M | Gemma 3 4B | 816x / **33x** | yes, on parseable replies |
| math v1 E-01 | 12,719,712 | Gemma 3 4B | 338x | no (at prior) |
| post-gen E-01 | 14,000,131 | Gemma 3 4B | 307x | fidelity only |

(With the language-model part only as the teacher size: 1,950x for the 6.1M tutor, 787x for the 4.9M reddit student.)

**Verdict on "100-20,000x" (thesis 3.1.2, new thesis.md:319, and 3.2 talking point line 351/357): NOT
supported as a claim about students that work.** The 20,000x end is the 0.6M coding-agent *pilot* student
that learned nothing; ~100x is the 103M/89M pilot students that also learned nothing. Supported ranges:
- agent students that clearly beat the baseline: **about 220x to 2,000x** (53.7M to 6.1M, Gemma 4 12B);
- all students that clearly learned the task, any domain: **about 30x to 2,000x** (reddit 129M to tutor 6.1M);
- all sizes trained (working or not): 33x to ~19,000x.
The old thesis already stated "34x-tól 860x-os méretcsökkentés" for the April reddit sizes (old thesis.md:204). HIGH.

**"1M-100M" (3.1.5 talking point, new thesis.md:338)**: trained sizes span 0.6M-129M; students that work
span 6.1M-53.7M (agents) and 4.9M-129M (reddit, April). Nothing at ~1M works; on the full tutor data the
size sweep stopped at 53.7M (xl ~100M was never trained on it: experiments/_common.py:26-32 lists xl, but
math_full.py's cloud loop trained xs/s/m/l only, running.txt:15-20). Safe: "students of a few million to
~50 million parameters (2.7M stayed at the baseline)". HIGH.

---

## 2. The agents

### 2.1 Math tutor, size results (first cloud loop, 64,554 train records, one seed each) - HIGH

results/math_full.csv; findings.txt F-M1 (263-278). Task accuracy final (best): xs 2.7M .050 (.050) | s 6.1M
kd .340 (.351) | s 6.1M ce .304 (.307) | m 16.4M .495 (.495) | l 53.7M .580 (.580); baseline .041 (362 answers).
Teacher-forced token accuracy .722/.788/.785/.824/.819; own-history token accuracy .219/.268/.260/.302/.304.
Best test CE m .782 < l .848 < s .959, yet l answers best. 53.7M still rising at pass 10. n=1 per size.
Trained on the A100 (5 side by side), not locally.

### 2.2 CE vs KL, 3 seeds, size m (16.46M), 195,980 train records, 10 passes - HIGH

results/math_agent.csv (pass 10 rows); recomputed here with `library/tooling/significance.py`:

| Metric (pass 10) | KL seeds 1/2/3 | KL mean (sd) | CE seeds 1/2/3 | CE mean (sd) | Paired KL-CE | Paired t (2 df), p | Welch (framework), p |
|---|---|---|---|---|---|---|---|
| Task accuracy | .700/.719/.697 | .705 (.012) | .684/.688/.677 | .683 (.006) | +.022 (sd .008) | 5.11, p = .036 | t 2.86, df 2.9, **p = .067** |
| Test CE (lower = better) | .481/.476/.486 | .481 (.005) | .528/.518/.527 | .524 (.006) | -.043 (sd .003) | -21.97, p = .002 | t -9.57, p = .0007 |
| Teacher-forced token acc. | .871/.873/.870 | .871 (.002) | .865/.869/.866 | .867 (.002) | +.005 | 6.24, p = .025 | p = .037 |
| Top-20 overlap with teacher | .384/.384/.379 | .382 (.003) | .352/.348/.350 | .350 (.002) | +.032 | | |

- KL ahead of CE at all 30 paired tests (10 passes x 3 seeds) on task accuracy (counted from math_agent.csv).
- The pairing is by design: the same seed gives both arms the same start weights and data order (commit 3e89b4e).
- The paired t-test in running.txt:96-97 is computed by hand; the framework's significance module only has
  Welch + Bonferroni, which gives p = .067 for the task-accuracy gap. Say "paired over 3 seeds".
- Notable: the KL student has a *lower test cross-entropy* than the CE student, which is trained on exactly that loss.
- Earlier n=1 direction: kd (0.9->0.1) vs ce at 6.1M, .340 vs .304 (13 of 362 answers), kd ahead on all 10 passes (F-M1).

### 2.3 Generalization drops (G-1) - HIGH for the numbers, n = 1 student

findings.txt:385-394 (experiments/math_generalization_check.py; math-agent-kl-m-s2, first-turn problems,
greedy, student's "Answer:" vs the *true* answer, 300 sampled per set):

| Set | 1 step | 2 steps | All |
|---|---|---|---|
| Test (held-out families, seen shapes) | .905 (222) | .404 (57) | .763 |
| Held-out shapes (add>mul, sub>div, sub>add) | .922 (102) | .169 (183) | .430 |
| SVAMP (human-written) | .433 (224) | .079 (76) | .343 |

Never-seen 2-step shapes .40 -> .17; SVAMP drops even on one step (.91 -> .43); 3.1% of SVAMP tokens are
outside the input vocabulary; Gemma's own SVAMP accuracy was not measured (UNVERIFIED baseline). One seed.

### 2.4 Gold-history vs own-history

- Token level (exists, HIGH): eval_epoch `teacher_forced_accuracy` (Gemma's text as context) vs
  `student_accuracy` (the student's own tokens fed back, position-wise match with Gemma's reply). CE-vs-KL
  m students: .871 vs .380 (KL, 3-seed means) and .867 vs .378 (CE); first loop: .722-.824 vs .219-.304.
  The own-history number is strict: a correct reply worded differently counts as mismatched positions.
- Task accuracy is "own reply, gold conversation history": the test records carry Gemma's earlier turns.
- Conversation-level own-history (the student's own earlier turns in its context) is **not measured** for
  any working student. It exists only for the slice trial (results/slice_trial.csv, experiments/slice_trial.py:135-184),
  where everything is 0 (floor). UNVERIFIED / not done.
- Route accuracy and explanation quality of the *student* are not reported anywhere; the tutor's task
  accuracy counts only replies with an "Answer:" (library/domain/math_tutor/__init__.py:66-74, ExactAnswerMetric).

### 2.5 Coding agent - what exists and what does not (HIGH)

Exists:
- 20 base projects (16 Python, 4 HTML; 12-15 files, 372-490 lines each: counted in library/domain/code_agent/projects/),
  3,240 randomized versions, 12,960 requests (16 train x 190 + 4 test x 50 versions, 4 requests each; running.txt:128-130).
- Gemma 4 12B (bf16) played every request through the two-step harness: 12,938 of 12,960 kept (running.txt:64-71).
- Teacher data: 25,783 records (24,188 train / 1,595 test; scope 12,938, explain 6,610, find 6,235), 806 batch
  files, 18 GB, output vocabulary 11,007 tokens, fingerprint 63558d75 (batches/code_agent/google_gemma-4-12B-it/info.json).
- Gemma's quality on the 96-request price test: scopes 96/96, true lines covered 81-89%, places hit 108/149,
  decision right 94/96 (thesis_work/cloud_runbook.md:79-88).
- Pilot students on 36-48 records: all sizes write nothing usable (0 scopes); a memorisation check (300 updates
  of 4 on the pilot records) reproduces explanations word for word, so the student can learn the task on
  6,000-token inputs (running.txt:161-165). Expected with so little data.
Does not exist: any student trained on the full v1 data, any coding-agent student result, any CPU/laptop timing.
The plan (running.txt:48-50) was CE vs KL at ~20M or a size sweep, "time a local sample first": not started.

---

## 3. Reduced vocabulary

### 3.1 Sizes (HIGH unless marked)

| Domain | Output vocabulary (logit columns) | Input vocabulary actually used | Source |
|---|---:|---:|---|
| Reddit sentiment | 525 (64 / 113 / 193 in the vocab sweep) | 68,328 (E-01); 262,144 full in the April scaling runs | runs/e01-reddit-*/info.json; ideaA runs |
| Math v1 | 528 | 262,144 (full; reduced 1,282 only computed, never trained) | runs/e01-math-*/info.json; old thesis.md:219 |
| Post-generation | 26,659 | 262,144 (full; reduced 72,133 only computed) | runs/e01-postgen-*/info.json; old thesis.md:138, 219 |
| Math tutor (full run) | **19,502** (fingerprint 7bcb36cd) | 19,854 (first loop); 20,116 (all 204,808 records, incl. the unknown row) | batches/math_tutor_full_v1/.../info.json; runs/math-full-*, math-agent-* |
| Math tutor (slice) | 18,772-19,052 | 18,855-19,289 | runs/slice-*/info.json |
| Coding agent (v1) | **11,007** (output_tokens.json lists 10,883 tokens; the rest are the shared whitespace/prompt/aux sections) | not built (no student yet): UNVERIFIED | batches/code_agent/.../info.json; library/domain/code_agent/output_tokens.json |

The "18,357" in CLAUDE.md and thesis_positioning.html is stale (the slice's older file). The full-run students use 19,502.

Coverage of the teacher's own replies by the task-derived vocabulary (a proxy, not probability mass):
math tutor 204,395 of 204,808 replies kept, 413 dropped for a token outside the vocabulary (0.2%; the second
loop reused the first loop's vocabulary); coding agent 25,783 of 25,783 kept (batches/*/info.json
kept_by_turn_type / skipped_by_turn_type). HIGH.

### 3.2 Parameter arithmetic (HIGH: from info.json model_info + library/model/transformer.py)

Architecture facts: input `nn.Embedding(input_vocab, hidden)` (transformer.py:106) and a separate output
`nn.Linear(hidden, output_vocab)` with bias (transformer.py:118): **untied**. Body = attention (4 h^2 per
layer, no bias, transformer.py:310-314) + gated-GELU FFN (transformer.py:382-388) + RMSNorms.

Representative working student, math-agent-*-m (hidden 256, 6 layers, 4 heads):
- input embedding 20,116 x 256 = 5,149,696
- output head 19,502 x 256 + 19,502 bias = 5,012,014
- body: attention 1,572,864 + FFN 4,718,592 + norms 3,328 = 6,294,784
- total 16,456,494 (matches info.json); embedding + head = 61.7% of the model even after the reduction.
With Gemma's full 262,144-token vocabulary at the same hidden size and body:
- untied (as the student is built): 262,144 x 256 + 262,144 x 257 + 6,294,784 = **140,774,656 (8.55x larger;
  embedding + head = 95.5%)**
- tied (as Gemma ties them): 262,144 x 256 + 262,144 bias + 6,294,784 = 73,665,792 (4.48x).
Same arithmetic for the other working sizes: 6.1M (h 128) -> 68.4M untied (11.2x; embedding+head 82.8% -> 98.5%);
53.7M (h 512) -> 302.3M untied (5.6x).
Old thesis has the same kind of table for a 80h/5L body (42.72M -> 6.02M reddit, 0.66M math, 8.44M post-gen;
old thesis.md:142-148, 190; grafikonok/data/02_parameter_allocation.csv). F-68a: the 3.4M reddit student is
96% input embedding. Use as context, not a novelty claim (findings.txt:59-63 says so explicitly).

### 3.3 Teacher probability mass kept inside the reduced vocabulary

- What exists: one ledger note, "both sharp and soft domains retain ~99% of mass at their operating vocabulary"
  (thesis_work/idea_ledger.txt:34-38, idea-ledger-new.txt:106-114; committed 2026-08-26 in d76f950).
  Gemma 3 4B. No script, no CSV, no per-domain numbers, and which "soft" domain is not stated. LOW.
- Not measured for Gemma 4 12B or either agent. The in-vocabulary mass is not logged anywhere in the data
  generation code (no logsumexp over the full row in library/; thesis_positioning.html item 11 recommended
  adding it before the cloud run; it was not added). The user's memory note lists it as a to-do for submission.
- How the teacher's distribution is used: only the reduced columns are stored
  (library/shared/vocabulary.py:213-218 `extract_logits_as_vector`), and the softmax over them in the loss
  (library/training/losses.py:29-32) renormalises the teacher over the reduced vocabulary. Teacher entropy is
  also over the renormalised reduced vocabulary at T=1 (findings.txt:317-318, T-L3).
- New thesis 2.5.1 (thesis.md:262) "Egy saját mérés szerint ... kb. 99%-a ... a kimeneti szótáron belül
  maradt" rests only on that ledger note. Safe wording in section 8.

---

## 4. Cost and hardware

### 4.1 Rented GPU cost (MEDIUM: estimates = hours x hourly price in running.txt / cloud_runbook.md; no invoice read)

| Run | Card, price | Time | Cost | Source |
|---|---|---|---|---|
| Math loop 1: Gemma 10,176 conversations + teacher data 67,058 replies (82 GB) + 5 students | A100 SXM 80 GB, $1.79/h | Gemma 9.7 h, teacher data 3.8 h, students 7.2 h | ~$40 (of which ~$13 is student training) | running.txt:17-20; findings.txt:274-275 |
| Math loop 2: 20,821 more problems + scoring (204,808 records in all, 265 GB) | RTX PRO 6000 96 GB, $2.49/h | ~18 h | ~$45.50 (+ a 288-problem timing sample) | running.txt:32-39; cloud_runbook.md:62-74 |
| Coding agent: 12,960 requests + 25,783 records (18 GB) | RTX PRO 6000, $2.49/h | 15.6 h | ~$39 (+ a 96-request price test) | running.txt:64-70; cloud_runbook.md:79-88 |
| Total | | | **~$125 plus small extras (samples, CPU pods for downloads, ~$0.6/day volume storage)** | |

Credit trail (running.txt:25, 44, 70): ~$59.5 left after loop 1, $55 budgeted for loop 2, then "$50 added,
$62 in all", ~$23 left at the end. Consistent with ~$125-135 spent. "Kevesebb mint 150 dollár" holds as an
estimate; it is not checked against RunPod billing (UNVERIFIED by invoice; RunPod was not queried).
Note: the ~$40 of loop 1 includes training five students, so "teacher data cost" alone is ~$110-115.

### 4.2 Local training on the RTX 4070 (HIGH: summed `time_seconds` of every train_batch in the logs)

| Runs | Batch time per student | Speed | Wall |
|---|---|---|---|
| math-agent-kl/ce-m-s1..s3 (6 students, 10 passes, 61,260 batches each) | 12.7-14.1 h each (+ ~10 tests x ~3.5 min) | 0.75-0.83 s per batch of 32, two side by side | 3 seed pairs ~40 GPU-hours on the 4070 (spread over 29-44 h of wall time each, paused while the user was away) |
| math-full-* (5 students) | 4.0-6.8 h each | 0.72-1.21 s per batch | A100, 7.2 h wall side by side (not local) |
| slice / E-01 students | minutes | | local |

So "a tanulómodellek tanítása többnyire lokálisan ... néhány tíz óra alatt" holds for the CE vs KL study
(~40 h); the size-sweep students trained on the rented A100. Old thesis.md:325: a 500k-example reddit student,
3 epochs, ~6.5 h on the 4070.

### 4.3 Student size and memory; phone / laptop

- Checkpoint files (fp32 weights only): 16.46M student 65,853,248 bytes (66 MB); 53.7M student 214,967,046
  bytes (215 MB) (runs/math-agent-kl-m-s1/checkpoints, runs/math-full-kd-l-s1/checkpoints). In bf16 / int8 they
  would be ~33/16 MB and ~107/54 MB (arithmetic, not measured).
- **Phone: nothing measured anywhere in the repo** (grep for phone/latency/onnx/gguf/tokens per second finds
  only ideas and an earlier review recommending a CPU/phone table, thesis_positioning.html item 15). UNVERIFIED.
- **Laptop: nothing in the repo either.** I measured CPU-only inference myself (scratchpad/work/cpu_timing.py,
  GPU hidden with CUDA_VISIBLE_DEVICES=-1, a desktop Intel i7-13700K, PyTorch, KV-cached decoding, four
  one-step/two-step word problems through the real harness turn `demo.take_turn`, route + solution):
  - math-agent-kl-m-s2 (16.5M): 1 thread 0.34-0.57 s per turn, ~220 generated tokens/s; 4 threads ~300 tokens/s.
    All four answers right.
  - math-full-kd-l-s1 (53.7M): 1 thread 1.0-1.5 s per turn, ~85 tokens/s (answers right, but it invents
    pupil names, "Kiri"/"Ravi", in its fact lines).
  - Process memory ~1.0-1.25 GB, dominated by Python + PyTorch + the Gemma tokenizer, not the weights.
  A desktop core is faster than a phone core, so this supports "runs on a CPU without a GPU in well under a
  second per answer" and makes a phone *plausible*, but does not show it. Not repeatable from the repo
  (the script is in the scratchpad). MEDIUM.
- Coding agent: no student, and its inputs are 4,815-7,200 tokens (running.txt:165; html ~7,200), so any
  laptop claim for it is UNVERIFIED.

---

## 5. Framework features (exist, file:line, and test evidence)

All 157 unit tests pass on the CPU (run 2026-11-03). One line each.

| Feature | Where | What it does | Tests |
|---|---|---|---|
| Task-derived reduced output vocabulary | library/shared/vocabulary.py:142-203 (sections example / whitespace / prompt / auxiliary); agents build the example section from the teacher's own replies (library/domain/math_tutor/agentic.py, library/domain/code_agent/vocabulary.py) | the student's head is only the tokens the task needs | test_agent_layer.OutputVocabulary; test_data_gen.ValidatesStoredStepTokens |
| Teacher logits stored over it, renormalised in the loss | vocabulary.py:213-218 (one indexed copy of the full row); losses.py:28-32 (softmax / log_softmax over the reduced columns) | KL to the teacher restricted to the subset | test_training.LossTests (every divergence vs float64); test_data_gen.ReducedLogitsInOneCopy |
| Vocabulary fingerprint | library/shared/batch_files.py:42-45 (sha1 of the ordered token list); written into every batch (data_gen/saving_handler.py:15, 30) and info.json (generator.py:48, 77); generator refuses to append with a changed vocabulary (generator.py:88-91); trainer refuses a batch from another vocabulary when it is first read (training/batch_handler.py:161-171); loader checks a finished run (model/student.py:94-99) | stops KL targets landing on the wrong tokens after a vocabulary rebuild (this happened: X-C1c students can no longer be read, findings.txt:359-360) | test_data_gen.VocabularyFingerprint (4 tests incl. resume with a changed vocabulary refused); test_training.SplitTests.test_batches_from_another_vocabulary_are_refused |
| Generated vs teacher-forced teacher modes | data_gen/model_handler.py:173 (generate) and :313 (score_training_examples: the teacher reads a given target in one pass); per-record mode generator.py:58-59, 109-145; input_handler.py:36-38 | the agents' data = Gemma's own replies read back (teacher_mode "forced", 204,808 + 25,783 records) | test_data_gen.PerRecordMetadata (forced vs generate; answer_correct None for forced) |
| Per-record metadata | input_handler.py:44-48 (split, item_id, turn_type, group); generator.py:136-152 (kept / skipped per turn type) | training splits by label, not file order; per-turn-type counts in info.json | test_data_gen.PerRecordMetadata, JobCompletesWhenEveryRecordWasProcessed; test_training.SplitTests |
| Teacher setup recorded per data file | agent/runner.py:52-76 (bf16 on >= 30 GB, NF4 below; <name>.setup.json; a different setup is refused, runner.py:63-68) | one file never mixes NF4 and bf16 replies | test_code_agent.TeacherSetup (2 tests) |
| Refusal to resume from a mismatched state | training/trainer.py:90-110 (state.json step vs resume checkpoint step and batches per epoch; else RuntimeError) | no silent training from the wrong weights | test_training.RunTests: test_stale_or_missing_resume_checkpoint_is_refused, test_mid_epoch_stop_resumes_exactly, test_hard_crash_resumes_from_the_last_epoch_end |
| Majority baseline + excess per eval | training/evaluator.py:34-45 (from confusion-matrix row sums), :159-163 (logged as majority_baseline, task_accuracy_excess); trainer.py:315 (printed) | every task accuracy is read against "always the most common answer" | test_training.MajorityBaselineTests; test_tooling.MajorityBaselineTest (3), LogRecordTest |
| Seed cohorts, Welch + Bonferroni | tooling/run.py:378-409 (Experiment.stats / compare), tooling/analysis.py:120 (cohorts = tag + configuration minus seed, completed runs only); tooling/significance.py:25-43 (stdlib Welch, incomplete-beta p, Bonferroni) | e.g. results/e01_anneal_direction_pairs.csv (50 comparisons) | test_tooling.CohortTest (3), SignificanceTest (2). No paired test in the module (see 2.2) |
| Gold- vs own-history accuracy | training/evaluator.py:140-141, 178 (teacher_forced), 198 (rollouts) - token level, every eval; conversation level only in experiments/slice_trial.py:135-184 | the gap = cost of compounding errors | test_training.ModelTests (incremental decoder = full recomputation; batched rollouts = one at a time) |
| KL-ratio and temperature annealing | training/schedules.py:35-58 (one cosine start->end for kl_ratio, student and teacher temperature; optional KL cycle); losses.py:62-72 (optional entropy-weighted per-token ratio); divergences forward KL / reverse KL / JS / TVD losses.py:45-60 | e.g. math-full-kd-s1 logs kl_ratio 0.9 -> 0.500 at mid-run -> 0.1 | no dedicated schedule unit test (RunTests train with 0.9->0.1; LR warm-up tested: test_warmup_applies_to_the_first_batch...) |
| Custom student network contract | model/custom_student.py:1-20 (contract), :48-66 (_check_contract: output shape, no look-ahead), :86 build_student; library/specs.py:47-48, 77 (TrainingRun(model=..., model_settings=...)) | any network trains through the same loop; recorded in info.json, rebuildable | test_training.CustomModelTests (4) + RunTests.test_custom_model_trains_resumes_and_loads_back |
| Math tutor harness (checks + one re-ask) | domain/math_tutor/harness.py:1-30 (route by first tag, solution checked and asked again once, explanation checked and asked again once, out_of_scope answered by the harness :48-49, :142-143); checks.py:161 check_lines, :189 check_explanation | the agent = student + harness | test_agent_layer.HarnessWithAFakeModel, ExplanationCutOff, JunkBeforeTheRouteTag, ConcessionCheck, FactsMustBeTheNumbersUsed, etc. |
| Coding agent harness | domain/code_agent/harness.py:1-12 (scope checked and asked again once; context narrowed; explanation checked for made-up names, asked again once); checks.py:24 check_scope, :103 check_explanation | two-step find/explain | test_code_agent.ScopeChecks (6), ExplanationChecks (3), Harness (6: a bad scope is asked for again once, ...) |
| Domain-free agent layer | agent/runner.py:1-8, 87-102 (StudentAgentModel), 105 run_sessions | the same harness drives Gemma or a student | via the harness tests |
| Template bank | agent/question_templates.py:1-22 (engine: kind, route, focus, requires, depth, slots); domain/math_tutor/questions/bank/*.jsonl = 3,461 lines, 43 excluded (questions/excluded.jsonl) -> 3,418 | follow-up messages filled from the shown solution | test_agent_layer (DepthAfterOffTopic, WhatIfCandidates, Number/label transfer tests) |
| Custom AST interpreter (never exec) | domain/math_tutor/functions.py:1-10 (docstring), :61 Interpreter, :205 execute (assignments, + - * // %, whole numbers, state reads, one dict return; anything else NotAllowed) | answers come from executed code | test_items.FollowUpsWriteTheirOwnAnswer (4), ValidatorCases (24 broken items caught), WholeNumberArithmeticOnly |
| Modularity | domain/registry + 8 domain packages (domain/__init__.py:20-26: reddit, math_word_problem, post_generation, reddit_comment_with_feedback, long_context_recall, code_recall, math_tutor, + code_agent); domain/base.py:20-86 (the Domain contract) | new domain = one package | test_data_gen.TinyDomain (a domain defined inside a test) |
| Other reliability | never-wipe CSV writes (experiments/_common.write_rows), batch numbering from state.json, atomic batch writes (batch_files.py:156) | | test_tooling.WriteRowsTest (2); test_data_gen.CrashResumesAfterSavedBatches (4) |

Size of the code (wc -l, .py only): framework core ~6,050 lines (shared 1,747, data_gen 709, training 1,227,
model 889, tooling 1,189, agent 295); domains: reddit 132, post-gen 171, math v1 726, math tutor 4,813,
coding agent 1,233 (+ the 20 projects). So "a domain is a module" holds, but an agent domain is a substantial module.

Two precision points for 3.1.6:
- The student *does* use the teacher's tokenizer (model/student_vocabularies.py:18 loads the teacher's
  AutoTokenizer; the reduced vocabulary is a subset of Gemma's token ids, vocabulary.py:185-188). The
  difference from the toolkits is the teacher's *full vocabulary / head*, not the tokenizer.
- No on-policy distillation, no hidden-state distillation, no cross-tokenizer teacher (literature_review.html
  "What this thesis does not do").

---

## 6. Lessons that count as findings, stated honestly

### 6.1 Part 1 students at the majority baseline (X-C1) - HIGH
findings.txt:296-307, 324-333. E-01 reddit students predicted the most common label 93-100% of the time;
best .621-.630 / final .598-.620 vs baseline .626 (best epoch = epoch 0 in 29/50 runs). Math v1: true/false
96/132 (always "True" gives 87), numeric 5/284. So the E-01 reddit/math "KL = CE" null says nothing about
KL vs CE at low entropy, and the per-field numbers equal the label shares. Lesson: report the baseline and the
excess (the evaluator logs both since 2026-10-08). The April 500k reddit runs are the exception (section 1.2).
Data-limited: the same pattern appeared on the tutor slice until the data grew (X-C1b-d; F-M1).

### 6.2 KL vs CE: two domains with real teacher entropy, two kinds of evidence - HIGH
- Post-gen (teacher ~0.99-1.0 nat; exp(H) 2.7), E-01, 3 seeds, 14.0M students: final top-20 teacher-student
  overlap pure KL .305 > CE->KL .298 > flat .291 > KL->CE .284 > pure CE .264, seed sd .001-.002; mean target
  rank 694 / 743 / 769 / 806 / 1,071 (e01_anneal_direction.csv; F-E1). Welch + Bonferroni: 8/10 pairs
  significant; CE->KL vs flat (p_corr .081) and CE->KL vs pure KL (.069) are not (e01_anneal_direction_pairs.csv).
  This is fidelity to the teacher's distribution (teacher-forced), not generation quality; structural validity
  is .975-.999 on every arm and own-history token accuracy ~.045.
- Math tutor (teacher .17 nat overall), 3 seeds, 16.46M: pure KL beats pure CE on task accuracy by +2.2 points
  (paired p = .036; Welch p = .067), on test CE by -.043 (p = .002), and on top-20 overlap .382 vs .350.
- Reddit / math v1: no evidence either way (floor effect, 6.1).
Honest statement: where the teacher is uncertain, KL transfers more of the teacher's distribution than CE
(post-gen, tutor), and on the tutor this carries over to a small but consistent task gain (3 seeds).

### 6.3 The data curve (-0.34 per doubling) - HIGH numbers, MEDIUM strength
results/slice_curve.csv: 576 / 1,344 / 3,136 / 5,383 train replies -> test CE 2.9645 / 2.5596 / 2.179 / 1.8554
(one kd student per point, 5.9M, 10 passes, same 688-reply test set). Least squares on log2(replies):
slope **-0.339 per doubling** (se .013; 95% CI -0.39 to -0.29 with t(2 df); R^2 .997). Segments: -0.331,
-0.311, -0.415 per doubling: no flattening, the last segment is steepest. Caveats: one seed per point, four
points, one size; it is test *loss*, task accuracy stayed at 0.8-2.4% final (best up to 3.7%) below the 6.1%
baseline; the four students can no longer be re-read (vocabulary rebuilt, findings.txt:359-360). The later
full-data results (64,554 replies -> .495 at 16.4M; 195,980 -> .68-.72 at 16.5M) differ in student size,
loss, vocabulary and test set, so they are not points on the same curve. The figure script:
thesis_work/grafikonok/plot_11_math_tutor_data_curve.py.

### 6.4 Teacher entropy sits in the explanations - HIGH
findings.txt:398-400 (experiments/teacher_entropy_by_reply_type.py; nats per step, T=1, renormalised reduced
vocabulary): math tutor explain .323 / solve .038 / route .003, all .1725 (200 random batch files); coding
agent explain .242 / scope .043 / find .010, all .1725 (all test records). Part 1 for comparison
(results/domain_entropy.csv, 256 examples): reddit .0203 (exp 1.021; per field .020-.089), math v1 .0028,
post-gen 1.006 (exp 2.73). One teacher per domain, so tokenizer granularity is not a confound within a domain
(F-70a applies across teacher families only).

### 6.5 Two more measured lessons worth one line each
- Capacity orders the tutor students: 2.7M stays at the prior, 6.1M -> 53.7M .340 -> .580, 53.7M still rising (F-M1).
- Format before content: slice students wrote valid facts-first replies (98-100% valid) with invented objects
  and numbers (running.txt:231-236; X-C1d) - the same "right format, wrong values" as part 1's 100% validity vs
  ~60% values.

---

## 7. Old thesis and old HTML docs: what can strengthen 3.1-3.3

Old thesis = thesis_work/thesis.md (Hungarian, ~April-September 2026, part 1 only).
- Honest positioning sentence (thesis.md:24): "A dolgozat hozzájárulása nem egy teljesen új modellarchitektúra
  vagy egyetlen általánosan optimális tanítási eljárás, hanem ezeknek a részben ismert és kutatott technikáknak és
  módszereknek a több domainen vizsgált, gyakorlati kombinációja." Good model for 3.3's tone.
- Positioning list (thesis.md:71-75): standard elements / less standard (dynamic KL/CE as curriculum) / new
  combination (task-specific input and output vocabulary reduction + autoregressive domain output).
  Update: the annealed KL/CE is *not* new (literature_review.html claims check: Clark 2019, Peng 2025, Zhou 2024).
- Compression statement (thesis.md:204): "5M-tól 129M paraméterig ... 4,3 milliárd ... 34x-tól 860x-os
  méretcsökkentés" - verified (4.30B / 129.19M = 33x; / 5.27M = 816x; / 4.93M = 872x).
- Table 1 (thesis.md:134-148): input 68,328 / 1,282 / 72,133; output 525 / 528 / 26,659; 80h/5L model 42.72M
  -> 6.02M / 0.66M / 8.44M; logit values per example ~20.4k / ~16.9k / ~1.68M; batch ~5.9 / 5.5 / 512 MB.
  Caveat: the math 1,282 and post-gen 72,133 input vocabularies were computed, never trained (thesis.md:219).
- Table 4 (thesis.md:253-264): full vs reduced input vocabulary at similar budgets (5.27M vs 4.93M: 86.00 vs
  86.33%; 10.56M vs 10.13M: 86.06 vs 87.20%; 34.67M vs 36.79M: 86.84 vs 87.22%), one seed, different depth/width:
  "the freed embedding budget can go to the body". Usable as a small part-1 result with the parseable-only caveat.
- Cost data (thesis.md:324-325): Gemma 3 4B on a 20 GB A100 slice ~3 s per example (500k -> ~417 h); a
  500k-example student trains 3 epochs in ~6.5 h on the 4070. Supports "teacher data is the bottleneck".
- Table 2 (thesis.md:161-172, 8k examples): CE / KL / anneal within seed noise - but at the majority baseline
  (64.6% vs ~62.6%; X-C1). Do not reuse as evidence that KL = CE at low entropy.
- thesis_positioning.html (2026-10-08, adversarial review, sources opened that day): feature table idea
  for 3.1.6 (item 10: "TRL's GKDTrainer wraps SFTTrainer around pretrained HF models with on-policy generalised
  JSD; DistillKit offers offline distillation from compressed stored logits with KL/JSD/TVD; EasyDistill covers
  black- and white-box KD ... None of their documentation shows: from-scratch tiny students with a task-derived
  output head; domain plug-ins; distribution metrics logged during eval; or an agent harness whose training
  records come from the same Conversation class it runs with."); "What does not survive" list (VocabTailor
  for task-derived vocabularies, MathCAMPS for solver-checked what-ifs, Kim & Rush for soft labels on gold,
  NeMo Guardrails for out-of-scope, LLM4Teach for annealed KL/CE, DistillKit for offline logits); revised
  novelty sentence (positioning "Revised novelty sentence"); the phone counter-argument: Gemma 3 270M INT4
  used 0.75% of a Pixel 9 Pro battery for 25 conversations (Google blog [17] there) - so do not argue that
  phones cannot run 270M; argue size and from-scratch control instead.
- literature_review.html (73 references, each opened 7-8 Oct 2026; BibTeX in thesis_work/literature.bib):
  a ready **positioning table** (TextBrewer, torchdistill, TRL GKDTrainer, DistillKit, EasyDistill, KDFlow,
  MiniLLM/GKD/DistiLLM, pre-training distillation (Peng 2025; Gemma 2/3), TinyStories/TinyGSM, Structured Agent
  Distillation, Kang 2025, STAR, TinyAgent/Octopus v2, LLM4Teach, Harness-Zero) against seven features
  (from-scratch student, reduced vocabulary, stored teacher logits, annealed KD/CE, built-in distribution
  analytics, agent/harness distillation, verified data); a "Claims check" table (defensible / reword / do not
  claim as new); a "What this thesis does not do" list (no on-policy KD, no pretrained-student baseline, no
  cross-tokenizer teacher, no hidden-state distillation, hand-written harness, no real pupils, no constrained
  decoding, single-GPU scale). It does not cover torchtune or NeMo/Minitron (other agents' job).
- novelty_check.md (2026-10-07, ~120 tool calls): closest works table; TinyGSM random-init ablation (53.1 vs
  63.3 at 125M; 55.5 vs 65.9 at 350M; whether it includes the verifier is not verified); correction
  2026-10-08: VocabTailor (arXiv 2508.15229) builds task-derived static output heads (training-free), so only
  "a from-scratch head trained with KL to the teacher renormalised over the task subset" survives.
- Caution on an old reference: old thesis [11] "Samenko, I., et al. (2024). Fast Vocabulary Transfer ...
  arXiv 2402.09977" is not in literature.bib and its author attribution looks wrong (UNVERIFIED; check before reuse).

---

## 8. Overclaim audit, new thesis chapters 1-4 (thesis_work/new_thesis/thesis.md)

| # | Line | Current claim (short) | Evidence | Verdict | Safe wording (English) |
|---|---|---|---|---|---|
| 1 | 319 (3.1.2); 351, 357 (3.2 TP) | students "akár százszor-húszezerszer" (100-20,000x) smaller | 20,000x = 0.6M coding pilot that learned nothing; ~100x = 89-103M pilots that learned nothing (1.3) | NOT supported | "The students that learned their task are about 220-2,000 times smaller than their teacher (53.7M to 6.1M parameters against Gemma 4 12B's ~12 billion; about 30-870x for the part-1 sentiment students against Gemma 3 4B)." |
| 2 | 18 (1.1) | a maths agent that "akár egy középkategóriás okostelefonon is offline elfut" | no phone or CPU measurement in the repo; my desktop-CPU check: 16.5M student 0.3-0.6 s per answer on one core, 66 MB of fp32 weights | NOT measured | "...a maths agent whose distilled student has 16.5 million parameters (66 MB) and answers a word problem in under a second on a single CPU core without a GPU, small enough that offline use on a phone is plausible (not tested)." |
| 3 | 18 (1.1) | "...vagy akár egy középkategóriás laptopon is lokálisan futtatható kódmagyarázó ügynök" | no coding-agent student trained on the full data; pilots learned nothing | NOT supported (today) | "...and a code-search and explanation agent whose harness and teacher data are complete (the teacher plays the agent in it); its distilled student is [future work / reported in chapter 9 if trained]." |
| 4 | 9 (abstract) | "öt domaint vizsgál részletesen, és ezek közül kettőnél ... önálló ügynököket hoz létre" | 5 domains with teacher data and students, but the coding agent has no working student; reddit/math v1 E-01 students at the prior | Partly | "The thesis uses two teachers on five domains, and for two of them builds an agent from a model and its own harness; on the maths agent the distilled student works inside the harness, on the coding agent the teacher-side pipeline is complete." |
| 5 | 338 (3.1.5 TP) | "at 1M-100M parameters" | trained 0.6M-129M; working 4.9M-129M (reddit, April) and 6.1M-53.7M (agents); 2.7M at baseline | Overstated range | "students of a few million to about 50 million parameters (the agent); the smallest that clearly learned the task has 6.1M." |
| 6 | 372 (4.1) | teacher data of both agents "kevesebb mint 150 dollárnyi bérelt GPU-időbe" | ~$40 + ~$45.50 + ~$39 = ~$125 by hours x price (running.txt); includes ~$13 student training; no invoice checked | Holds, as an estimate | "The rented GPU time for both agents' teacher data came to about $125 (estimated from run times and hourly prices; this also includes training five students on the A100)." |
| 7 | 372 (4.1) | students trained "többnyire lokálisan, egy RTX 4070 ..., néhány tíz óra alatt" | CE vs KL: 6 students x 12.7-14.1 h, two at a time (~40 GPU-hours); the size sweep ran on the A100 (4-7 h each) | Holds, with a caveat | "The CE vs KL students trained on a local RTX 4070 (about 13-14 hours each, two at a time); the size-sweep students trained on the rented A100 (4-7 hours each)." |
| 8 | 372 (4.1) | "megtartsuk a tanármodell domain-specifikus tudásának nagy részét"; "a méretéhez képest kifejezetten jó teljesítményt" | best 16.5M student: ~70% of test answers match Gemma's (baseline 4%); 1-step .91 but 2-step .40, unseen shapes .43, SVAMP .34 (G-1); no same-size comparison model exists | Overstated | "...keeps a large part of the teacher's behaviour on problems like those it was trained on (about 70% of the teacher's answers on held-out problem families, against a 4% baseline), but far less on unfamiliar problem shapes and human-written problems." Drop "for its size" or add a baseline. |
| 9 | 380 (4.2) | the coding agent "képes ... megtalálni és ... megmagyarázni bizonyos kódrészeket" | true for Gemma in the harness (scopes 96/96, decisions 94/96 on the price test), not for a distilled student | Ambiguous | "...an agent design (model + harness) that, with the teacher as its model, finds and explains code in a 12-15-file project; distilling it is the next step." |
| 10 | 381 (4.2) | "offline, ingyenesen, akár telefonon is futtatható ... ügynök"; coding agent "laptop videokártyáján pillanatok alatt" | framed as "Képzeljünk csak el" (a vision); nothing measured | OK only as a vision | Keep it explicitly hypothetical: "One could imagine ..." and add "(not measured in this thesis)". |
| 11 | 7 (abstract) | "a domain definiálását követően mindössze konfigurálással beállítható ..." | true once a domain module exists, but an agent domain is 1,233-4,813 lines of domain code | Holds with a caveat | "Once a domain module is written, data generation, training and the student's architecture are set by configuration alone; the teacher data can be reused for many students (11 students on the maths agent's data so far)." |
| 12 | 21 (1.2), 373-377 (4.1) | framework covers data generation, training, evaluation; fine control; detailed analysis; modular | all present (section 5), 157 tests pass | Supported | Keep; optionally cite the features in section 5 and the test count. |
| 13 | 262 (2.5.1) | "Egy saját mérés szerint ... kb. 99%-a a kimeneti szótáron belül maradt" | one ledger note (idea_ledger.txt:34-38), Gemma 3 4B, no script or numbers; never measured for the agents | Weak | "An early informal check on the part-1 domains (Gemma 3 4B) found about 99% of the teacher's probability inside the output vocabulary; it was not repeated for Gemma 4 12B and the agents, where only 0.2% of the teacher's replies needed a token outside the vocabulary." |
| 14 | 265 (2.5.2) | "A kódolóügynöknél hasonló, mesterséges intelligenciával bővített feladatadatbázis ..., a korábbi programatikus ... képest" | the coding data IS programmatic: 20 base projects varied by a seeded program into 3,240 versions; requests from topic phrases (code_agent/project.py:1-20, requests.py:1-12) | Inaccurate | "For the coding agent, 20 base projects (written with AI help) are varied by a program (renamed names, reordered functions, files and page sections) into 3,240 versions, whose true places are found again by parsing." |
| 15 | 274 (2.5.4) | "Ugyanaz a 16,4 millió paraméteres tanulómodell ... 49,5% ... 195 980 ... 68-72%" | same size (m) but different runs: kd 0.9->0.1 at 16.39M vs pure KL / pure CE at 16.46M, different input vocabulary and test set (362 vs 3,147 answers) | Imprecise | "A student of the same size (16.4M) solved 49.5% of the test problems after 64,554 training replies, and 68-72% after 195,980 replies (pure KL or pure CE, three seeds each); the loss settings and test sets differ, so this is not a controlled comparison." |
| 16 | 271 (2.5.4) | "-0,34 per doubling ... a javulás még nem lassult le" | slope -0.339, CI -0.39..-0.29, 4 points, one seed each, loss not accuracy | Holds with a caveat | add: "(one student per data size; the task accuracy stayed below the most-common-answer baseline at these sizes)". |
| 17 | 280 (2.5.5) | 4.9M reddit student: "az értékek 86%-át találta el, szemben ... 64%" | over the 8,210 parseable of 10,016 test comments (April code skips unparseable replies); ~71% if those count as wrong; one seed; pre-fix code | Holds with a caveat | "...86% of the values in the 8,210 replies it could parse (out of 10,016), against 64% for always giving the most common label." |
| 18 | 388 (4.4) | reddit: "KL- és CE-veszteséggel is viszonylag hasonló feladatpontosság volt elérhető, hiszen ilyen éles eloszlásnál a két veszteség gyakorlatilag ugyanazt a célt adja" | in those comparisons the students sat at the majority baseline (X-C1), so the similarity is a floor effect | Misleading as an explanation | "KL and CE students reached similar task accuracy here, but in these comparisons (2,048-8,000 examples) the students had not got beyond always giving the most common label, so the comparison cannot show whether the sharp distribution makes the two losses equivalent." |
| 19 | 389 (4.4) | "minél nagyobb volt a KL aránya a tanítás végén, annál jobb volt az egyezés"; "a különbség nem a véletlen műve" | means are ordered; 8/10 pairs significant after Bonferroni; CE->KL vs flat and CE->KL vs pure KL are not | Mostly holds | "...the overlap grew with the KL share at the end of training; 8 of the 10 pairwise differences stay significant after Bonferroni correction (3 seeds each)." Also say it is fidelity, not quality. |
| 20 | 268 (2.5.3) | "ha egy módszer jól működik 4, illetve 12 milliárd paraméternél, akkor feltételezhetjük, hogy nagyobb modelleket is desztillálhatunk hasonlóan" | untested; capacity-gap work (Zhang et al. 2025 in literature_review.html) says a bigger teacher is not automatically better for a tiny student | Assumption | "We expect the method to carry over to larger teachers, but this is not tested here, and a larger teacher does not automatically make a better tiny student." |
| 21 | 346 (3.1.6 TP) | toolkits "assume the student uses the teacher's tokenizer" as a contrast | our student also uses Gemma's tokenizer (student_vocabularies.py:18) | Wrong contrast | "...assume the student keeps the teacher's full vocabulary and output head (and is usually a pretrained model)". |

Claims checked and supported as written (no change needed): Gemma 4 12B "közel 12 milliárd", ~24 GB bf16
(2.1.1, line 35); entropies 0.02 / 0.003 / 1.0 / 0.17 / 0.17 nat and 0.24-0.32 vs ~0.04 (2.2, lines 87, 110,
122, 153, 227, 233); the 224-conversation, 1,345-reply trial: 0/54 vs 4/54 baseline (2.5.4, line 271); the
rude 2/129 and friendly 16/56 counts (line 280); "A100 és RTX PRO 6000", bf16 on rented GPUs (lines 372, 385);
post-gen overlap 30.5% vs 26.4%, seed sd 0.1-0.2 pp, validity above 97% (line 389); 20 projects, 16 Python + 4
HTML, 12-15 files (lines 156, 380); NF1 (line 293: no overwrite, vocabulary and resume refusals) - all in section 5.
