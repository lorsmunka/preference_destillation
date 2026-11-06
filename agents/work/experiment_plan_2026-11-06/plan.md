# Experiment plan for thesis chapter 9 (2026-11-06 → data freeze 2026-11-09 noon)

Written for the agent that will orchestrate the runs. Ranked by importance (how much the result strengthens a
chapter 9 claim) against cost. Numbers on timing come from thesis_work/running.txt; check them on the first pass.

## Rules that bound the budget
- **Local GPU** (RTX 4070, 12 GB): only while the user is present; pause with the STOP file when they leave, resume
  by rerunning the same script. Realistic local time until the freeze: ~30-34 GPU hours. Two students side by side
  each run ~0.86 s/batch instead of ~0.70 (keep both windows visible: hidden windows slow ~35-50%).
- **Rented GPU**: may run unattended. Credit left ~$23 (running.txt, 2026-10-28; check the balance before booking).
  Gemma itself is measured only in bf16 on a rented card, never the local 4-bit copy.
- Single numbers = the final pass; KL vs CE = seed-matched pairs; task accuracy = agreement with Gemma's answer.
- The test set is the fixed tail of the batch folder, so `max_training_examples` subsets keep the same test.

## The plan, by importance

| # | Experiment | Supports (chapter 9 claim) | Cost | Importance |
|---|---|---|---|---|
| 1 | **Coding agent: KL vs CE in the learning phase (epochs 1-3).** Resume `experiments/code_agent_ce_vs_kl.py` for CE only to the end of pass 3, then STOP (it would go on to 10). KL's test loss is lowest at pass 2, so later passes only show overfitting. Already now: CE is worse than KL at pass 1 (2.946 vs 2.904) and pass 2 (2.881 vs 2.811). | KL vs CE on long inputs, in the phase where the model still learns | ~40 min local | High |
| 2 | **Coding agent split by step**, evaluation only: the KL m checkpoint (runs/code-agent-kl-m-s1) on its test records, loss and token accuracy separately for `scope` and `explain`/`found` records, at pass 2 (best loss) and pass 10. Small script; no training. | Where the coding student fails: locating code in ~6,000 tokens vs explaining from the narrowed lines | ~15 min local + script | High |
| 3 | **Math tutor end to end**: the full-data KL student (math-agent-kl-m-s1) through the harness (`run_sessions` + `StudentAgentModel` + `TemplatePupil`) on the held-out test conversations (377 in tutor_chat), compared turn by turn with Gemma's saved conversation (conversations/full_v1.jsonl): route agreement, final-answer agreement on the task turns (same task text), re-asks, harness catches. Pupil follow-ups depend on the shown solution, so compare per event type, not per message. | The *agent* works, not just single replies; the cost of compounding errors (own vs teacher history) | ~1 h local + script | High |
| 4 | **Gemma 4's probability inside the reduced vocabulary**, both agents: Gemma 4 12B bf16 on a rented card (an RTX A6000 at ~$0.59/h did the attention probes in ~8 min), teacher-forced over ~500 test replies per agent, full softmax, share of the mass on the output vocabulary. | The reduced vocabulary loses little (2.5.1 only cites Gemma 3 4B on sentiment, ~99%) | ~$0.5-1 rented, unattended | High |
| 5 | **Math data curve on one test set**: KL m (same settings and seed as math-agent-kl-m-s1) with `max_training_examples` = 25% (48,995) and 50% (97,990), side by side. With the 100% runs that gives 3 points on the same 8,415-reply test. | Data amount matters more than size; a clean curve (today's points use different test sets) | ~8-9 h local (both together) | High |
| 6 | **KL → CE seed 3** (`experiments/math_ce_vs_kl.py`, the kl-to-ce arm). | The schedule result on 3 seeds instead of 2 | ~13 h local (~16 h if side by side) | Medium |
| 7 | **Size on the full data**: KL at size s (6.1M), one seed, side by side with #6. | Size vs data at the full data, next to m | ~7-9 h local | Medium-low |
| 8 | **Coding agent with half the projects**: KL m on 8 of the 16 training projects, 3 passes. | Does fewer projects mean earlier overfitting? (data variety) | ~1 h local | Medium-low |
| 9 | **Paraphrase test**: Gemma bf16 answers the test rewordings (rented), then the student; agreement with Gemma's answers. | Generalization to reworded problems | ~$1 rented + ~0.5 h local | Low |
| 10 | **Size l (53.7M) on the full data**: needs the 265 GB of math data uploaded to a rented volume and ~13 h of an A100 (~$25-30 with the volume). | Size at full data, large end | over the credit | Only if money is added |

## Suggested order
- **Nov 6 (today):** #1 (40 min) → #2 (15 min) → #3 (~1 h). Book #4 on a rented card in parallel (unattended).
- **Nov 6 evening → Nov 7:** #5, both runs side by side (~9 h of presence; pause overnight, resume in the morning).
- **Nov 7 → Nov 8:** #6 and #7 side by side (~16 h of presence over two days).
- **Nov 8:** #8 if there is a free hour; #9 only if #4 went smoothly and credit is left.
- **Nov 9 noon:** data freeze. Every run that has not finished is reported with the passes it has, or left out.

Budget check: local ≈ 0.7 + 0.3 + 1 + 9 + 16 + 1 ≈ 28 h of ~30-34 available; rented ≈ $1.5-2 of ~$23.

## Deliverables per experiment
A results CSV through `experiments/_common.write_rows` (never open a results file with "w"), the run directories,
and a short line in running.txt's STATUS block. Report final-pass numbers and every pass for curves.
