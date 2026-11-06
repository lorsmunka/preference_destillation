# Chapter 9 synthesis (2026-11-06)

Written by the synthesis agent (CPU only; helper scripts and their outputs are in this folder:
`dump_runs.py` -> `runs_info.csv`, `runs_evals.csv` (every eval_epoch scalar from the 205 runs); `paired_math.py`;
`gemma_code_quality.py`; `code_attention_passes.py` -> `code_attention_passes.csv`, `gemma_metrics.txt`;
`math_l_passes.py`). Saved to disk by the main session, because the agent could not write files.

Conventions: "pass N" = the eval_epoch record with epoch index N-1; single numbers are pass 10 unless marked.
Task accuracy = agreement with Gemma (`library/training/evaluator.py:123-124` passes `example['model_response']`;
math = same `Answer:` (`library/domain/math_tutor/agentic.py:75-81`); coding = same canonical scope
(`library/domain/code_agent/__init__.py:46-51`)). [baseline] = majority-answer facts, not for thesis wording.
Strength: S = several seeds and a consistent direction; W = one seed or a pilot; F = floor effect;
D = data or teacher only, no student.

## Summary
1. **Math tutor, KL vs CE (3 seeds):** KL is ahead at 30/30 seed-matched passes on task accuracy, test CE, top-20
   overlap, mean target rank, top-20 hit and teacher-forced accuracy. Final paired gaps: +2.2 pts task accuracy
   (t 5.1), -.043 CE (t -22), +3.2 pts overlap (t 15), rank -4.9 (t -78).
2. **Own-history token accuracy does not separate the arms:** KL ahead at 25/30 passes, final +0.2 pts (t 0.7), CE
   ahead in seed 3. The KL gain is in the final answer and in following Gemma's distribution, not in word-by-word
   agreement.
3. **KL->CE:** level with KL up to pass 6, then behind; test CE worse than KL from pass 5 on. Ending on CE brought
   nothing.
4. **The coding agent memorises:** train CE falls to .54 (train token accuracy .835) while test CE rises from 2.81
   (pass 2) to 3.05, and only 2/800 test scopes match Gemma's. Test token accuracy and overlap still rise to pass
   10, so most of the loss rise is overconfidence (student entropy 2.35 -> .97).
5. **Gemma's own scopes** match the true places exactly in only 372/800 test requests (line recall .879, precision
   .824).
6. **Coding attention per pass (new):** the student moves its attention from the project to the request (x4 -> x23
   an even spread) and to its own reply. It never weights the target lines above an even spread (lift .94 -> .86).
   Its best head finds the target file in <=5/23 probes; Gemma's best head finds it in 23/23. The drift is steady and
   stops after pass 8; nothing turns at the loss minimum.
7. **The loss choice leaves no mark on attention** (seed .598 vs loss .386). The 53.7M math student's attention is
   flat after pass 8 while its test loss turns up. Bigger first-loop students are *less* Gemma-like (.336 -> .205)
   but answer better.
8. **Part 1:** E-01 reddit and math v1 are floor effects. The post-gen fidelity ordering holds on 3 seeds. Sentiment
   at ~490k comments did learn its labels (86% over parseable replies), one seed per size.
9. **Draft issues:** 6.2 uses best-epoch reddit numbers; the 7.1 "validity" label is wrong; the coding vocabulary is
   10,883 in the text vs 11,007 in the trained students; `attention_models.csv` lost Gemma's math and code rows (key
   collision); G-1 is scored against the true answer and uses the best seed; 4.4 and 10 present a floor effect as a
   property of the loss.
10. **Highest-value additions:** conversation-level own- vs gold-history accuracy for the full-run students (~2-4 h,
    CPU possible); the coding student in its harness at pass 2 vs pass 10, plus a "swap the project" context test
    (CPU, <1 h each); KL->CE seed 3 (~13 h local); a data curve on the 16.5M student (~6 h); retained probability
    and Gemma's SVAMP accuracy in bf16 (<$2).

## 1. Result inventory

### Part 1
| Experiment | Setup | Final-pass numbers | Strength | Source |
|---|---|---|---|---|
| Sentiment pilot (Oct 2025-Feb 2026) | 470M student failed (token encoding bug); then a 3 h pilot on 100 batches | 10-batch test: own-history token acc 87.69%, teacher-forced 98.97% | W | thesis_work/thesis.md:267; git 908f320 `evals/__talking_points.txt` (470,149,213 params) |
| Sentiment scaling (Mar-Apr 2026, HUN-REN vGPU) | 7 sizes, 5.3M-129M; ~490k train / ~10k test; 3 passes; KL .99->.3; 1 seed each | field-level task .860 (5.3M) ... .872 (53M) ... .874 (129M); own-history token acc .926 -> .933 | W | grafikonok/data/03_sentiment_scaling_curve.csv; saves/runs-archive/tsc-scale-*.tar.gz eval_epoch 2 |
| Reduced vs full input vocabulary | 3 pairs (~5/10/36M) | reduced .863/.872/.872 vs full .860/.861/.868 | W | data/04_reduced_vs_full_vocab.csv |
| Confusion matrices (fig 2.3) | 4.93M run | 86.34% of values vs 63.76% [baseline], over the 8,210 parseable of ~10,016 replies; equals the run's logged `classification_accuracy` (.8633), so the old metric left out unparseable replies | W | recount of data/08b_*.csv; archived info.json (500,000 examples, ratio .98) |
| E-01 reddit | 5 arms x 10 seeds; 3.4M; 2,048 comments; 3 passes; data order fixed | final .598-.620 vs .626 [baseline]; 0/10 pairs significant | F | experiments/results/e01_anneal_direction.csv; findings.txt:296-327 |
| E-01 math v1 | 5 x 10; 12.7M | .218-.236 vs .209 always "True" [baseline]; numeric 5/284 | F | same CSV; findings.txt:300 |
| Idea A and idea D | 3.4M, 5 seeds | excess <= 0 everywhere; each size had its own test set | F | thesis_work/results_first_pass.md |
| Post-gen E-01 | 5 x 3; 14.0M; 3,840 examples; 10 passes | top-20 overlap KL .305 > CE->KL .298 > flat .291 > KL->CE .284 > CE .264 (sd .0002-.0017); rank 694/743/769/806/1071; validity .975-.999 in every run; 8/10 pairs survive Bonferroni | S | e01_anneal_direction*.csv; findings.txt:327 |
| Post-gen top-k (Apr 2026) | 1 seed per arm | superseded by E-01 | W | postgen_kl_anneal_topk.csv |
| Virág | data only (4,968 examples) | teacher 99.88% vs the generator's truth; no student ever trained | D | findings.txt:239 |

### Part 2, math tutor
| Experiment | Setup | Final-pass numbers | Strength | Source |
|---|---|---|---|---|
| Slice trial | 5.9M; 1,345 replies | 0/54 answers; conversation-level own- and gold-history accuracy both 0 | F | slice_trial.csv |
| Slice data curve | 5.9M; 576-5,383 replies | test CE 2.96 / 2.56 / 2.18 / 1.86; answers .008-.024 vs .061 [baseline] | W (loss), F (answers) | slice_curve.csv |
| Copies vs new | 2 seeds | test CE 1.80 / +new 1.48 / +copy 1.59 / +both 1.39; digit gap 19 -> 27/26/28 pts | S (loss) | slice_mix.csv |
| First loop, size sweep | 64,554 train / 362 test answers; KD .9->.1 | xs 2.7M .050, s 6.1M .340, m 16.4M .495, l 53.7M .580, CE at s .304 | W; xs = F | math_full.csv |
| Full run, KL vs CE | 16,456,494 params; 195,980 / 8,415 replies (3,147 with an answer) | KL .700 / .719 / .697 (.705, sd .012); CE .684 / .688 / .677 (.683, sd .006) | S | math_agent.csv |
| Full run, KL->CE | 2 seeds | .697 / .704; test CE .498 / .492 | S (n=2) | runs/math-agent-kl-to-ce-m-s*/logs |
| G-1 generalisation | KL seed 2 only, scored against the **true** answer | test .763; held-out shapes .430 (2-step .169); SVAMP .343 | W | findings.txt:385-397 |

### Part 2, coding agent
| Experiment | Numbers | Strength | Source |
|---|---|---|---|
| Pilots (pilot_nf4, pilot2_nf4) | task 0; test CE 6.4-7.8 (~10 updates) | F | code_agent_trial_pilot*_nf4.csv |
| Gemma's full bf16 pass, test split | route right 778/800; line recall .879; precision .824; places hit 901/1,112; scope = true places exactly 372/800 | D | gemma_code_quality.py over conversations/v1.jsonl |
| KL m (11.95M; 24,188 / 1,595 records, 800 test scopes) | test CE per pass 2.904, **2.811**, 2.830, 2.881, 2.920, 2.970, 2.996, 3.027, 3.045, 3.052; teacher-forced acc .524 -> .590; train CE 3.62, 1.87, 1.43 ... .54 (train token acc .835); task .0025 = 2/800 | W | runs_evals.csv; train_batch records |
| CE m (stopped after pass 4) | test CE 2.946 / 2.881 / 2.913 / 2.971; KL had lower test CE at 4/4 shared passes (-.04 to -.09), better overlap 4/4, better rank 4/4 | W | runs_evals.csv |

Teacher reference points: Gemma answered right on 97.1% (second loop) and 94-97% (first loop) (running.txt:12, 36).
Teacher entropy on the test sets: math .1722, coding .1725 nats.

## 2. Math tutor in detail

Seed-matched comparison, pass 10 (`paired_math.py`):

| Metric | KL | CE | KL - CE per seed (s1 / s2 / s3) | t | KL better at (of 30) |
|---|---|---|---|---|---|
| task accuracy | .7052 | .6829 | +.016 / +.031 / +.020 | 5.1 | 30 |
| test CE | .4808 | .5240 | -.047 / -.042 / -.041 | -22 | 30 |
| teacher-forced token accuracy | .8713 | .8667 | +.006 / +.005 / +.003 | 6.2 | 30 |
| top-20 overlap | .3823 | .3499 | +.032 / +.036 / +.029 | 15.2 | 30 |
| mean target rank | 15.26 | 20.16 | -4.8 / -4.9 / -5.0 | -78 | 30 |
| own-history token accuracy | .3800 | .3776 | +.003 / +.008 / -.004 | 0.7 | 25 |
| student entropy (Gemma .172) | .420 | .345 | +.07 to +.08 | 50 | KL higher at 29 |
| termination before the length cap | .847 | .848 | ~0 | - | - |

- Task accuracy by pass (mean of 3 seeds): KL .107 .405 .505 .580 .611 .656 .677 .695 .702 .705; CE .099 .373 .477
  .566 .596 .635 .658 .677 .680 .683. Gains are under 1 point per pass from pass 9.
- About 15% of test replies in every arm hit the 2x length cap without a closing tag; which reply types is UNVERIFIED.
- `validity_rate` counts only answer-bearing replies that have a parseable `Answer:` (`library/shared/metrics/task.py:231-259`).
- KL->CE (seeds 1-2): task accuracy vs KL at pass 10 -.003 and -.015; test CE worse from pass 5, gap growing to
  +.018 / +.016; better than CE on loss 20/20 and task accuracy 19/20; overlap .3615, rank 16.3; the sharpest
  (entropy .338).
- Size (first loop, one seed): test CE xs 1.236, s .959, m .782, l .867. l's lowest test CE was .848 at pass 8, then
  it rose while task accuracy kept rising. Train CE at pass 10: l .236 vs m .390. Own-history token accuracy .219 /
  .268 / .302 / .304. KD beat CE at size s on all 10 passes, on task accuracy and test CE (n=1).
- Data: same m architecture .495 at 64k replies -> ~.70 at 196k (different test sets and schedules: a direction
  only). New problems beat renumbered copies at equal reply counts (10,262 vs 10,422).

## 3. Interpretability

### 3a. Math tutor (does not overfit: test CE falls through pass 10)
Source: experiments/results/attention_models.csv, rows math-{kl,ce}-m-s1[-passN].

| Pass | Gemma-likeness KL / CE | Attention entropy | Numbers when writing a number | Message | Previous token |
|---|---|---|---|---|---|
| 1 | .264 / .256 | .526 / .521 | .357 / .361 | .209 / .199 | .088 / .085 |
| 2 | .275 / .269 | .539 / .532 | .382 / .375 | .217 / .216 | .082 / .083 |
| 3 | .285 / .278 | .560 / .554 | .385 / .385 | .225 / .220 | .080 / .081 |
| 5 | .296 / .297 | .588 / .577 | .379 / .367 | .233 / .229 | .074 / .080 |
| 10 | .308 / .307 | .615 / .611 | .365 / .359 | .236 / .237 | .072 / .074 |
| untrained | .043 | .999 | .143 | .220 | .007 |

- Gemma on the math probes (mean over its 768 heads; recomputed in gemma_metrics.txt because the CSV lost the row):
  entropy .517, previous token .091, copy .043, message .119, reply so far .476, outside view .445; numbers when
  writing a number .178 mean head, .643 best head. Mean-head values are not comparable with a 24-head student.
- Gemma-likeness rises every pass in both arms; KL ahead by .004-.008 at passes 1-4, level from pass 5. Most of it is
  shared recency: with position removed .182 / .177 vs .089 untrained (attention_novelty_check.md C1).
- Number lookup peaks at pass 3 and then falls while task accuracy climbs .50 -> .70.
- Paired KL - CE at the final pass (3 seeds): entropy +.004/+.005/+.003, copy +.0007/+.0010/+.0014, look-back
  distance -0.4/-0.2/-0.3 tokens; same sign in all seeds but each under one seed sd: no loss effect.
- The seed dominates: matched-head correlation .386 same loss / other seed vs .598 other loss / same seed (bootstrap
  [.380, .394] vs [.592, .604]); loss effect .0036, permutation p = .11 (attention_seed_pairs.csv; running.txt:119-133).
- Size: Gemma-likeness xs .336, s .327, m .275, l .205 against task accuracy .05 -> .58; likeness does not track ability.
- Overfitting in attention (`math_l_passes.py`): the 53.7M first-loop student overfits by loss after pass 8. Its
  attention entropy .465 (pass 1) -> .238 (pass 4) -> .292 (pass 8) -> .301 (pass 10); number lookup flat .36-.37;
  message .189 -> .223 -> .225. Nothing changes when its loss turns up.

### 3b. Coding agent, attention per pass
`code_attention_passes.csv`: 24 scope probes, means over heads; the CPU rows for pass 8 reproduce the study's
`code-kl-m` row to 4 decimals.

| KL m pass | test CE | teacher-forced acc | message | reply so far | other files | target lines (lift) | best head line / file hits |
|---|---|---|---|---|---|---|---|
| 1 | 2.904 | .524 | .009 | .043 | .824 | .036 (.94) | 0 / 1 of 23 |
| 2 | 2.811 | .550 | .018 | .068 | .789 | .034 (.89) | 1 / 3 |
| 3 | 2.830 | .564 | .021 | .069 | .787 | .033 (.88) | 2 / 1 |
| 5 | 2.920 | .580 | .028 | .093 | .756 | .033 (.86) | 2 / 0 |
| 8 | 3.027 | .590 | .054 | .106 | .722 | .032 (.85) | 3 / 4 |
| 10 | 3.052 | .590 | .056 | .104 | .723 | .033 (.86) | 2 / 3 |
| even spread / untrained | - | - | .0024 | .0018 | .877 | .038 (1.00) | 0 / 1 |
| Gemma (bf16) | - | - | .045 mean (best head .48) | .270 | .482 | .026 mean; best head lift 9.4 | 16 / 23 and 23 / 23 (layer 36, head 15) |

- CE m passes 1-4: the same drift, slightly faster toward the request (message .012/.015/.027/.030), lift .94 -> .86.
- The pilot students' attention equals an untrained student's (lift 1.00, message .0024).
- Answer: the student learns to look at the request and its own reply, never at the target file or lines. The drift is
  steady from pass 1 to 8 and flat afterwards; nothing changes at pass 2 (lowest test loss), so overfitting has no
  distinct attention signature. The data cannot tell memorised request->scope patterns from retrieval not learned
  yet; a test that would: give the same request with another project version (or project) and check whether the
  scope changes (cheap CPU run on the existing checkpoints).

## 4. Figures
Existing, fit for chapter 9:
- `thesis_work/grafikonok/14_math_agent_kl_ce.png`: the main figure (the 2-point task gap is hard to see on 0-80).
- `experiments/results/math_agent.png`: superseded by 14.
- `11_math_tutor_data_curve.png`: already figure 2.2; refer back.
- `03_sentiment_scaling_curve.png`: part-1 lesson (saturation); one seed per size, old trainer, parseable-only metric.
- `09_postgen_topk_metrics.png`: one seed; replace with an E-01 3-seed chart.
- `05_structured_loss_spread.png`: not evidence that KL ~ CE (floor effect).
- Attention page FIG 1B, 3F, 5A, 6B (runs/attention_study/attention_study.html): need PNG export.

New figures with the data on disk:
1. Paired KL - CE per pass, one thin line per seed, task accuracy and test CE (math_agent.csv).
2. Top-20 overlap and mean rank per pass for KL / CE / KL->CE (runs_evals.csv): the strongest separation.
3. Size sweep: task accuracy and test CE per pass, 4 sizes (math_full.csv + runs_evals.csv): l's loss turns up while
   its answers improve.
4. Coding agent: train CE (train_batch) vs test CE and test token accuracy per pass.
5. Coding attention per pass vs Gemma (code_attention_passes.csv).
6. Optional: teacher exp(H) per domain next to the post-gen overlap gap (domain_entropy.csv).

## 5. Weak spots in thesis_work/new_thesis/thesis.md
(N#) = already in notes_before_submitting.md.

| Section | Problem | Evidence |
|---|---|---|
| 6.2 | reddit "62-63%" are best-epoch values; the final pass is .598-.620 | e01 CSV; findings.txt:298 |
| 4.4, 10 | "KL ~ CE on sentiment because the distribution is sharp": E-01 reddit is at the floor, and no CE arm exists at the 490k scale (N11) | X-C1; all tsc runs are KL .99->.3 |
| 10 | "memorised the 16 projects" is supported by the train/test gap (CE .54 vs 3.05; token acc .835 vs .590), but test token acc and overlap still rise to pass 10 and student entropy falls 2.35 -> .97: much of the loss rise is overconfidence | runs_evals.csv; train_batch records |
| 2.5.5 | 86% vs 64% counts parseable replies only (N3) | recount |
| 2.5.4 | "Ugyanaz a 16,4M": the full-run student has 16,456,494 params and is a different run (N11) | info.json |
| 3.3 vs 3.1.4 / 3.2 | "6-100M" vs "6-54M" (N13) | math_full.csv |
| 7.1 | "validity = feldolgozható válaszok aránya" is really the share of answer-bearing replies with a parseable `Answer:`; values (seed 1) verified; chapter 9 should own the 3-seed numbers | task.py:231-259 |
| 7.2 | coding output vocabulary 10,883 vs 11,007 output rows in the trained students (math's 19,502 matches) | runs/code-agent-*-m-s1/info.json |
| 6.5 | "470M ... 39 nap" is ~10 days of data generation + ~29 days of training | git 908f320 talking points |
| 6.1 vs the old TDK draft | 470M vs "~537M" | thesis_work/thesis.md:185 |
| 2.5.1 | the ~99% retained probability comes only from a Gemma 3 ledger note (N2) | idea-ledger-new.txt:106-112 |
| G-1, if used | scored against the true answer, not Gemma's; best seed (s2); Gemma's SVAMP accuracy never measured | math_generalization_check.py:1-6 |
| 6.3 | "data, not training length": the flattening coincides with the cosine LR decay, so it does not separate data from schedule | info.json (warmup .1 + cosine) |
| 6.3 | ">700 GB" UNVERIFIED | - |
| attention CSV | the `gemma` key collides across kinds in `write_rows`; the surviving row is the sentiment (Gemma 3 4B) one, mislabelled "Gemma 4 12B" | experiments/_common.py:71-94; attention_report.py:567-573 |
| findings X-C1c | "2.97" should be 2.96 (CSV 2.9645) | slice_curve.csv |

Results that appear more than once (chapter 9 should own them): the first-loop sizes (2.5.4, 6.2, 6.3); 70.5% / 730x
(3.3, 10); copies vs new (6.3); the post-gen overlap (4.4, 10); the coding overfit (10); the 7.1 values.

## 6. Experiments with the most weight before 11-11
| # | Experiment | What it shows | Cost | What exists |
|---|---|---|---|---|
| 1 | Own- vs gold-history accuracy at the conversation level, the 8 full-run math students | the agent-level number CLAUDE.md asks for; KL vs CE at conversation level | ~2-4 h GPU, or CPU at 0.3-0.5 s per turn | slice_trial.py:116-165 (teacher_solutions, history_accuracy); demo.load_student |
| 2 | Coding student in its harness at pass 2 and pass 10 + the project-swap context test | scope vs true places and vs Gemma; does it read the context at all | <1 h per checkpoint; CPU feasible | code_agent_trial.py report path; StudentAgentModel; code_agent/demo.py |
| 3 | KL->CE seed 3 | three seed pairs for the schedule claim | ~13 h local | math_ce_vs_kl.py |
| 4 | Data curve on the 16.5M student, one test set: 6k / 20k / 65k replies (196k exists) | replaces fig 2.2's confounded slice curve (N1) | ~6 h local | max_training_examples |
| 5 | Gemma 4 bf16 on a rented A6000: retained probability for both vocabularies + Gemma's own answers on SVAMP and held-out shapes | 2.5.1 on Gemma 4; G-1 restated as agreement with Gemma | ~$1-2 | the attention_study teacher pod pattern |
| 6 | Resume CE m on the coding agent to pass 10 | completes the coding pair | ~4 h | resume point kept |
| 7 | Attention per pass for math seeds 2-3 | n=3 for 3a | ~1 h CPU | code_attention_passes.py pattern |

Not worth doing before the deadline: a size sweep on the full data (tens of GPU hours).

## 7. Talking points for chapter 9

Math tutor
1. The 16.46M student, trained from scratch, gives Gemma's final answer on 70.5% of the 3,147 answer-bearing test
   replies (KL, 3 seeds, pass 10). (math_agent.csv)
2. KL beat CE at 30/30 seed-matched passes on task accuracy, test CE, overlap and rank; paired gaps +2.2 pts (t 5.1,
   2 df) and -.043 CE (t -22).
3. The gap is in the distribution (overlap +3.2 pts; rank 15.3 vs 20.2), not in own-history token agreement (+0.2 pts).
4. KL->CE gave no gain: level with KL until pass 6, worse loss from pass 5.
5. Both arms flatten after pass 8 (gain under 1 point per pass).
6. At 64k replies: 2.7M stays at the prior; 6.1 -> 16.4 -> 53.7M give .340 -> .495 -> .580. The 53.7M student's loss
   rises after pass 8 while its answers still improve: loss and task accuracy disagree.
7. 64k -> 196k replies: .495 -> ~.70 (different test sets). New problems beat renumbered copies at equal cost (2 seeds).
8. One seed, against true answers: never-seen 2-step shapes .17 and SVAMP .34 vs .76 on test.

Coding agent
9. Gemma's own scopes equal the true places exactly in 372/800 test requests (recall .879, precision .824).
10. The student matches Gemma's scope in 2/800. Test CE is lowest at pass 2 (2.81) and rises to 3.05 while train CE
    falls to .54; test token accuracy still rises (.550 -> .590).
11. CE m shows the same pattern; KL had lower test CE at 4/4 shared passes (n=1).

Interpretability
12. The loss leaves no mark on attention; the seed does (.598 vs .386).
13. Students grow more Gemma-like every pass, mostly through recency; the largest student is the least Gemma-like
    but answers best.
14. The coding student looks at the request (x23 an even spread) and its own reply, never at the target lines
    (lift <= .94); its best head finds the target file in <=5/23 probes, Gemma's best head (layer 36) in 23/23.
15. Overfitting has no attention signature: flat after pass 8 in both the coding student and the 53.7M math student.

Part 1 lessons
16. On sharp domains (exp(H) ~ 1.02), students trained on 2k examples learned only the class prior, so the KL/CE null
    there is a floor effect.
17. On post-gen (exp(H) 2.73) the arms are ordered by how much KL is in the late loss; 8/10 pairs survive Bonferroni
    (3 seeds).
18. With ~490k comments the labels are learned and size barely matters: 86.0 -> 87.4% from 5 to 129M (one seed each,
    parseable replies only).

Comparison with chapter 3 (no new literature)
19. 16.46M vs Gemma 4 12B is ~730x; the agent students in [39] and [40] are pretrained and start at 0.5-1.1B.
20. The output vocabulary is 19,502 of 262,144 tokens (7.4%), trained into the student, unlike VocabTailor's
    inference-time cut [48]; vocabulary is still 61.7% of the parameters (5.15M input + 5.01M output), vs MobileLLM
    125M at >20% [47].
21. KL (word level) and CE on Gemma's text (sequence level, [27]) both distil here; the measured edge of the
    distribution is +2.2 pts / -.043 CE. Hinton's 3%-data result [16] is context only, not like-for-like.
22. TinyAgent's 80.06% vs 79.08% for GPT-4-Turbo [40] is scored against ground truth; here the metric is agreement
    with the teacher, so a student cannot exceed the teacher by construction.
