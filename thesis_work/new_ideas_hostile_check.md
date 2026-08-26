# HOSTILE CHECK — three new ideas (+1 component scan), 2026-08-26

Same verdict key as `idea_ledger.txt`. Checked against literature, and against this
project's own measured numbers where those decide the question.

---

## N1. Gaussian noise on attention / embeddings to delay overfitting in tiny distills

**VERDICT: WOUNDED — mechanism owned at every injection point named; the causal story
that makes it work does not transfer to from-scratch students.**

Owned:
- **Embeddings**: NEFTune (2310.05914) is this idea verbatim for embeddings — inject noise
  into input token embeddings during training, reduce overfitting, large gains. A 2026
  follow-up (2605.23171) already refines it.
- **Attention**: AttentionDrop (2504.12088) is a *family* of stochastic regularizers on
  attention distributions, and one member, "Blurred Attention Smoothing", applies Gaussian
  convolution over attention logits. Attention dropout / DropHead / DropAttention predate it.
  Bayesian transformers use learnable Gaussian noise on pre-softmax logits.
- **In KD specifically**: noise-regularized teachers (SMART, 2310.01209), Noisy Feature
  Distillation, SoKD, Noisy Self-KD for summarization (2009.07032).

The kill-shot is not the citations, it is the mechanism. **NEFTune works because noise stops
the model overfitting to instruction-data specifics so it falls back on its PRETRAINED
knowledge.** A from-scratch 3.4M student has no pretrained prior to fall back on. The engine
producing NEFTune's headline number is structurally absent here, so the honest prior is null
or harm — not "it works well on small distills".

What genuinely supports the idea, from our own logs: these students **do** overfit in the
regime we would test — task accuracy decays 62.7% -> 59.2% across epochs at 2048 examples.
So there is real overfitting to attack; the question is whether noise is the right lever.

Counter-consideration that cuts the other way: measured teacher entropy on reddit is **0.021**
— the teacher is nearly one-hot, so KD is contributing almost no label-smoothing of its own.
That leaves more room for an injected-noise regularizer than a soft-teacher argument implies.

Surviving sliver: "does noise regularization transfer when the pretrained prior it relies on
is absent?" Same replication-at-the-floor species as S1-S4. TDK-fine, journal-weak.

---

## N2. Alternating loss steps (every 2nd/3rd step KL) vs ratio mixing / annealing

**VERDICT: DEAD ON ARRIVAL — it is the ratio mix, in expectation. Also re-enters ledger #1.**

This one does not need a citation to kill. Choosing KL with probability p and CE with 1-p per
step gives

    E[grad] = p*grad_KL + (1-p)*grad_CE = grad of the p-weighted mixed loss

Step-alternating at 66/33 has **the same expected gradient** as mixing at 0.66/0.33. It is not
a different objective; it is a higher-variance estimator of the same one. Searches turned up no
prior work comparing alternation to weighting in KD — but that absence is evidence the question
is near-trivial, not that it is unclaimed territory.

It also walks straight back into `idea_ledger.txt` #1 (loss-schedule family: Hinton,
Annealing-KD, CTKD, DKD; Menon ICML 2021 predicts CE~=KL), which we already shot down — and
our own runs already measured CE~=KL.

The **one** place a real effect could live: adaptive optimizers. Adam normalizes by a running
RMS of gradients, so alternating losses changes the second-moment statistics and is *not*
mathematically equivalent under Adam even though it is under SGD. That is a genuine, subtle
difference — but it is an optimizer-interaction question, not a distillation contribution, and
it belongs in an ablation column, not a thesis.

Do not spend a thesis on "annealing vs step-annealing".

---

## N3. Is MoE viable sub-200M when distilling from a dense teacher?

**VERDICT: PREMISE LARGELY REFUTED by existing scaling laws — but it hides the best surviving
question of the three.**

Direct evidence against, at exactly our scale: "Slicing and Dicing: Configuring Optimal
Mixtures of Experts" (2605.11689) reports **all MoEs underperform dense baselines at 10M and
20M active parameters** under Chinchilla-optimal data, and only overtake dense when given
roughly **100-400x tokens per active parameter** (~20x Chinchilla). Our students are 3.4M-14M
trained on 2k-50k examples — orders of magnitude away from where MoE starts winning. Joint MoE
scaling laws (2502.05172) and 2509.23678 tell the same story: MoE's win needs data and compute.

Our own ledger #10 already killed the adjacent idea (chain of tiny experts vs MoE on edge) as
SCOOPED + PREMISE REFUTED, because edge is memory-bound (2606.21428) and MoE must hold every
expert in memory while using a fraction — the worst possible trade on device.

The live research direction is the **reverse**: pruning/distilling MoE teachers into dense
students (2605.28207), and KD *for* MoE LMs (2502.12947).

**But**: every one of those scaling laws trains on hard next-token targets. Distillation gives
far more information per example — a full teacher distribution instead of one label. MoE's
documented weakness at small scale is precisely **data hunger**. So the sharp question is:

> Does distillation's higher information-per-example move the MoE viability threshold *down*
> — and if so, by how much?

That is mechanistically motivated, unanswered by the papers that set the threshold, and it has
a real headline either way ("soft targets buy MoE N-fold data efficiency" or "they don't, the
threshold is architectural"). It is the best surviving question in this batch. It is also the
most expensive to build.

---

## +1. Component scan (shallow, not systematic)

Already on `candidate_board.txt` cluster F: Canon (F2), gated attention (F1), hyper-connections
(F3), differential attention (F4), DSA (F6). Not yet on the board and worth a look:

- **Selective Attention** (2410.02703, Google) — parameter-free modification, improves
  transformers, ablated at moderate scale. Cheap to port; untested tiny/distilled.
- **2-simplicial attention** (2507.02754) — trilinear attention, a genuine new *component*
  rather than a schedule; efficiency framing is for large scale, so the scale-down question is
  open, but the kernel work is nontrivial.

Neither displaces Canon (F2) as the strongest scale-down piece, and the board's own note that
Canon's window is closing still stands.

---

## On "build better, more real-world domains first"

**This is the most expensive item on the list and it can damage the thesis's anchor.**

The one axis this project owns is the **reduced OUTPUT vocabulary** (~525 tokens). A richer,
more real-world domain has a larger output space — and we can already see what that does:
post-generation has **V = 26,659**, where the example section alone is 26,484 tokens, the
vocab-reduction premise mostly evaporates, and the `auxiliary_token_percentage` sweep knob
stops working entirely (aux% moves V only 26,505 -> 26,659). Making domains more real-world
trades away the unclaimed axis for a more crowded one.

What the experiments actually need is not "more real-world" but **more discriminative**.
Reddit saturates: teacher-forced 97-98%, top-k accuracy 1.00, mean target rank 1.0 — three
metrics pinned at ceiling, unable to separate any hypothesis. Math has a verifiable answer and
~22% accuracy, i.e. real headroom.

Cheaper and sharper than a real-world corpus: **one decidable formal task** — which is what
S6 in `shelf_workable.txt` already argues for. A task where the correct answer is provable
gives an unambiguous metric, free ground truth, and headroom, at a fraction of the cost of
curating realistic data.

---

## Ranked verdict for TDK / journal

| Idea | TDK | Journal | Note |
|------|-----|---------|------|
| N3 reframed (does KD lower the MoE data threshold?) | Strong | Plausible | Only one with a real headline either way; most build cost |
| N1 (noise at the from-scratch floor) | Adequate | No | Predicted-null; a clean negative is still defensible |
| N3 as originally posed (is MoE viable sub-200M) | Adequate | No | Confirms an existing scaling law |
| N2 (step-alternating loss) | Weak | No | Same objective in expectation; ablation column at best |

Against the shelved batch: **E1 (output vocab vs the softmax/rank bottleneck) still outranks
all three of these** on novelty-per-unit-cost, and it is already half-measured.
