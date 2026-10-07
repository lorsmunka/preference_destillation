# Review: tooling / analytics / domains / experiments (adversarial)

## 2026-10-08 start
Task: adversarial review of library/tooling, library/domain (base/registry/non-math cfg), analyze.py, experiments/*.py, math_tutor_report.py, domains_report.py, progress.py, library/shared/metrics, library/shared/logging.
Status: starting. Previous reviewer (stopped by API error) left scripts in scratchpad/review_tooling/ (cohorts, e01_verify, eval_consistency, ideas_verify, majority, math_breakdown, code_recall_*, confusion, lcr_check, metric_edges, ties, write_rows_test, progress_check, mtr_check, strict_check). No save file from them; results to be re-established.
New scratch scripts: C:\Users\Ors\AppData\Local\Temp\claude\D--preference-destillation\ffe199e2-eaba-41f3-879f-8ba3f4ec1e4f\scratchpad\review_tooling2\
Findings: none yet.

## 2026-10-08 step 1 (read tooling/run.py, analysis.py, render, logging, metrics, domains, _common, anneal_direction, report_ideas; re-ran e01_verify, ideas_verify, majority, math_majority, metric_edges, confusion)
VERIFIED C1 (critical, misleading tables): reddit task accuracy sits AT the majority-label baseline of each test set; no baseline is computed anywhere.
  Majority baseline (always predict per-field majority) on the BatchHandler test split: 256 ex -> 0.6133, 512 -> 0.6172, 1024 -> 0.6708, 2048 -> 0.6262, 4096 -> 0.6202.
  Students: ideaD 256 best 0.6078, 512 0.6164, 1024 0.6672; E-01 reddit arms best 0.621-0.630, final 0.598-0.620 (baseline 0.6262).
  confusion.py: share of predictions = majority label is ~1.0 for tone/safety/toxicity in most runs; ideaD-data01024-s2 epoch0 = 0.6708 exactly, ideaD-data02048-s3 epoch0 = 0.6262 exactly (constant output).
  => findings.txt "Data plateau 256 -> 60.8%, 1024 -> 66.7%" = the test set's class prior (0.613 -> 0.671), not learning. "safety 74 / tone 63 / toxicity 59 / sentiment 43" ~= test-set label shares (0.748 / 0.627 / 0.587 / 0.543). E-01 reddit null is a degenerate instrument.
  Math: 'True' is 20.9% of the 2048 test set; students 0.22-0.25 (math_breakdown run pending).
VERIFIED: e01_anneal_direction.csv matches fresh collect_rows (115 rows, 0 diffs) and RUNS configs match info.json (0 diffs). F-E1 post-gen numbers match the CSV.
CANDIDATE: F-E1 "no arm separates by more than one seed std on task accuracy" false on task_accuracy_final: reddit ce-to-kl .6199 vs pure-ce .5982 (2.2pp, sd 1.7-1.8); math ce-to-kl .2356 vs flat .2180 (1.8pp, sd .9-1.7).
VERIFIED stale CSVs: ideaA_vocab_sweep.csv V=113 row has runs=3 (5 completed now: best .6203+/-.0146 vs CSV .6206+/-.0186); ideaD_data_plateau.csv 2048 row runs=1 (3 completed now: best .625+/-.016 vs CSV .6028). findings "excess-over-chance 0.169 at V=113" computed from the n=3 row.
VERIFIED metric edge: ClassificationMetric._parse crashes (TypeError) when the student output parses as JSON but not an object: '"tone"', '5', 'null'. ExactAnswerMetric returns 'answer: 15' when the answer line is preceded by a newline (search starts at the \n).
Checked fine so far: Experiment.stats ddof doc'd (pstdev default); compare uses sample sd for d/Welch; RunLogs parse; EvalEpoch legacy field mapping; all info.json have training_test_ratio.
