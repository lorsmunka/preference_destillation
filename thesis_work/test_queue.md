# TEST QUEUE — ideas ranked by cost to test

Ranked 2026-08-26 against the code as it actually stands, not against the idea's
ambition. Sources: `idea_ledger.txt` (graveyard + survivors), `candidate_board.txt`
(clusters A–J), `shelf_workable.txt` (S1–S6, OQ1–3).

The ledger is mostly a graveyard, so ranking *it* alone would order dead ideas. What
follows ranks every idea across all three files that still has a testable claim.

## The finding that reorders everything

`auxiliary_token_percentage` is already a matched-body output-vocabulary knob:

| aux% | 0.0 | 0.1 | 0.25 | 0.5 | 0.75 | 1.0 |
|------|-----|-----|------|-----|------|-----|
| V (reddit) | 64 | 113 | 193 | 300 | 427 | 525 |
| V (math) | 50 | 104 | 184 | 297 | 426 | 528 |

Task-essential tokens (example + whitespace + prompt) are always retained; only the
auxiliary bank varies. That is an 8x vocab sweep at fixed capacity, on one domain,
with **zero framework changes** — the ANCHOR (A1–A4), OQ1 and S2 all wanted exactly
this. It does not reach post-gen's 26k end: there the example section alone is 26,484
tokens and aux% moves V only 26,505 -> 26,659.

## Tier 1 — runnable today, no framework change

| # | Idea | Ease | Promise | Value | Why now |
|---|------|------|---------|-------|---------|
| E1 | **A1–A4 + OQ1 + S2 — output-vocab vs softmax/rank bottleneck** | Easy (config) | High | **Highest** | Board's only hard-pass survivor. The V sweep is a config axis today; gradient-norm and isotropy are measurable post-hoc from checkpoints via forward/backward hooks. |
| E2 | **G1 — KL/CE ratio as controller of the self-consistency dividend** | Easy (inference) | Med-High | High | `generate()` already samples with temperature; needs only a voting harness. Board calls it the cheapest item; a null is publishable. |
| E3 | **B1/B2/B3 — structural validity vs value accuracy** | Easy (log analysis) | Med | Med-High | Already instrumented: `validity_rate` + per-category confusion are logged every epoch. Calibration run shows validity 1.00 against task 0.63 — a 37pp gap sitting there. |
| E4 | **C1/C2 — calibration of tiny reduced-vocab students** | Easy (analysis) | Med | Med-High | ECE computes in the analysis script from logits; rides on E1's runs, so it costs no extra training and adds a second axis to the anchor bundle. |
| E5 | **D3/D4 — data-efficiency / synthetic-data plateau** | Trivial (config) | Med | Med | Pure `max_training_examples` sweep. Lowest novelty of the five, but free and it locates the floor every other experiment sits on. |

## Tier 2 — small framework change first (all four approved 2026-08-26)

| Idea | Blocker | Size |
|------|---------|------|
| F2 Canon layers, F1 gated attention | `TrainingRunner` hardcodes `Transformer(...)`; no `model=` on `TrainingRun` | Small |
| A1–A4 at full range (525 -> 26k) | No direct output-vocab-size knob; needs mass-ranked truncation of the example section | Medium |
| S1 co-compression | No prompt-internalization on/off switch | Medium |
| A2 gradient-norm *curve* | Trainer logs no gradient norms (post-hoc snapshots work without it) | Small |

## Tier 3 — not worth compute

`#15` reduced input vocab (mathematically inert), `#20` vocab-as-safety (DEAD),
`D1` logit storage (demoted, owned), `G2` schema-masked KL (own retained-mass probe
predicts null), `J2`/`J3`/`J4`, `F3`/`F4` (confirmatory / N-A), `#14` knob-turns.

## Measurement notes from the calibration run

- 2048 examples x 3 epochs = **173s/run** on the RTX 4070; task accuracy 0.63 vs 0.81
  at 50k, so 2k sits well above the signal floor. 128 examples does **not** — validity
  and task accuracy are both 0.00 there.
- `topk_accuracy` and `mean_target_rank` saturate (1.0 and 1.0) at V=525 with k=20:
  at these vocab sizes the target is essentially always in the top 20. Read fidelity
  from `teacher_student_overlap`, `student_entropy`, `perplexity` and KL loss instead.
- Task accuracy peaks at epoch 0 and decays (0.627 -> 0.592) at 2k examples: these runs
  overfit, so report the best epoch as well as the last.
