# Experiment plan for thesis chapter 9 (2026-11-06 → data freeze 2026-11-09 noon)

Written for the agent that will orchestrate the runs. Ranked by importance (how much the result strengthens a
chapter 9 claim) against cost. Revised after the synthesis report (agents/work/chapter9_synthesis_2026-11-06/report.md,
sections 5-7); read that report first. Timing figures come from thesis_work/running.txt; check them on the first pass.

## Rules that bound the budget
- **Local GPU** (RTX 4070, 12 GB): only while the user is present; pause with the STOP file when they leave, resume
  by rerunning the same script. Realistic local time until the freeze: ~30-34 GPU hours. Two students side by side
  each run ~0.86 s/batch instead of ~0.70 (keep both windows visible: hidden windows slow ~35-50%).
- **Rented GPU**: may run unattended. Credit **$27** (user, 2026-11-06: "I don't mind using it").
  Gemma itself is measured only in bf16 on a rented card, never the local 4-bit copy.
- Single numbers = the final pass; KL vs CE = seed-matched pairs; task accuracy = agreement with Gemma's answer.
- The test set is the fixed tail of the batch folder, so `max_training_examples` subsets keep the same test.

## Already done (no run needed)
- **Coding agent KL vs CE in the learning phase:** CE m has 4 passes (paused, STOP in place); KL had the lower test
  CE at 4/4 shared passes (-.04 to -.09), better overlap and rank at 4/4. KL's test loss is lowest at pass 2, so
  finishing CE to pass 10 adds only overfitting; leave it paused.

## The plan, by importance

| # | Experiment | Needed to say | Cost | Importance |
|---|---|---|---|---|
| 1 | **Coding student, where it fails**: (a) the KL m checkpoints at pass 2 and pass 10 on the test records, loss and token accuracy split into `scope` vs `explain`/`found` records; (b) the student in its harness (`code_agent/demo.py`, `StudentAgentModel`) on the test requests: scope vs Gemma's and vs the true places; (c) **project swap**: the same request with another project version or project, does the scope change? | whether the student locates code at all or only replays request→scope patterns (memorising vs retrieval not yet learned, report §3b) | <1 h each, CPU possible | High |
| 2 | **Math tutor end to end, all 8 full-run students** (3 KL, 3 CE, 2 KL→CE): own- vs gold-history accuracy at the conversation level, through the harness on the held-out test conversations, compared with Gemma's saved conversations (conversations/full_v1.jsonl). Reuse `slice_trial.py:116-165` (`teacher_solutions`, `history_accuracy`) and `demo.load_student`. | the agent works as a whole; the cost of compounding errors; KL vs CE at conversation level | ~2-4 h GPU, or CPU at 0.3-0.5 s per turn | High |
| 3 | **Gemma 4 bf16 on a rented card** (the attention-study pod pattern; an RTX A6000 at ~$0.59/h did its probes in ~8 min): (a) the share of Gemma's probability inside each agent's output vocabulary, teacher-forced over ~500 test replies per agent; (b) Gemma's own answers on SVAMP and the held-out shapes. | 2.5.1 on Gemma 4 (today only Gemma 3 on sentiment); the G-1 generalisation numbers restated as agreement with Gemma | ~$1-2 rented, unattended | High |
| 4 | **Math data curve on one test set**: KL m (settings and seed of math-agent-kl-m-s1) with `max_training_examples` = 6k, 20k and 65k replies; with the 196k runs that gives 4 points on the same 8,415-reply test. | how test loss and agreement grow with data, on one test set (replaces the confounded slice curve of figure 2.2) | ~6 h local | High |
| 5 | **KL → CE seed 3** (`experiments/math_ce_vs_kl.py`, the kl-to-ce arm). | the schedule result on 3 seeds | ~13 h local, or ~$5-8 on a rented card overnight (needs the 265 GB of math batches uploaded, ~1.5-2 h, volume ~$0.7/day) | Medium |
| 6 | **Attention per pass, math seeds 2-3** (the `code_attention_passes.py` pattern). | n=3 for the math attention table (report §3a) | ~1 h CPU | Medium-low |
| 7 | **Coding agent with half the projects**: KL m on 8 of the 16 training projects, 3 passes. | whether fewer projects means earlier overfitting | ~1 h local | Low |
| - | Size sweep on the full data | | tens of GPU hours | Not before the deadline |

## Approved by the user (2026-11-06)
All seven, with two notes:
- **#4 on a rented card if it comes out cheap.** Best paired with #5: one upload of the math batches (265 GB,
  ~1.5-2 h; #4 alone would need only the first ~65k records' files plus the test tail, ~96 GB) and one card running
  #4's three runs and #5 side by side overnight. Rough guess ~$8-10 of card time at ~$0.59/h plus ~$3 of volume.
  Time one pass and price from the live catalog first; if it does not come out cheap, run #4 locally (~6-8 h).
- **#5 stays on the math tutor.** The user asked about KL → CE on the coding agent instead: not worth it now. The
  coding student learns only in passes 1-2 and overfits after, so a schedule that moves toward CE over 10 passes
  would mostly act during the overfitting; KL vs CE there is already answered (KL lower test CE at 4/4 passes).
  Math seed 3 is what takes the schedule claim from 2 seeds to 3.

## Suggested order
- **Nov 6 (today):** #1 (CPU, <3 h in all) and #2 (GPU while the user is here, else CPU). Book #3 on a rented card.
- **Nov 6 evening → Nov 7:** #4 locally (~6 h of presence). Decide #5: local (~13 h of presence over Nov 7-8) or
  rented overnight (upload first; time one pass, price from the live catalog before committing).
- **Nov 8:** #6 on CPU; #7 if there is a free hour.
- **Nov 9 noon:** data freeze. Every run that has not finished is reported with the passes it has, or left out.
- Delete every pod and volume when done (the cloud scripts stop themselves; test that first).

Budget check: local ≈ 3 + 6 (+13 if #5 stays local) h of ~30-34; rented ≈ $1-2 (#3) + $5-8 and ~$3 of volume if #5
moves to a rented card: ~$10-13 of $27.

## Deliverables per experiment
A results CSV through `experiments/_common.write_rows` (never open a results file with "w"), the run directories,
and a short line in running.txt's STATUS block. Report final-pass numbers and every pass for curves.
Known bug to fix before reusing the attention CSV: the `gemma` key collides across kinds in `write_rows`
(experiments/_common.py:71-94, attention_report.py:567-573), so Gemma's math and code rows were overwritten.
