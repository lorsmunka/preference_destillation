# LAP 5 — attention / mech-interp direction: questions, pitfalls, contradictions

2026-08-26. Scope caveat: ~5 searches plus one paper fetch. Enough to find leads and
kill bad ones; NOT an exhaustive novelty search. Treat verdicts as provisional.

---

## CONTRADICTION 1 (in our own notes) — the attention-sink argument that killed F1

`candidate_board.txt` cluster F says:

> "F1 Gated Attention = UNDERCUT/likely-null: short seq -> ~0% attention sinks
> (2504.02732, 2603.05498), so sink-elimination is moot; kill the sink sub-claim"

**"When Attention Sink Emerges in Language Models: An Empirical View"** (2410.10781,
ICLR 2025) points differently:

- "attention sink emerges in small LMs, **even in Pythia-14M**"
- they train custom **60M models from scratch** and watch sinks appear "between 1k and 2k"
  optimization steps
- sinks are "largely induced to facilitate **short-context** prediction"; restricting
  training to long sequences makes the sink ratio **collapse**
- BUT "smaller window size prevents the emergence of attention sink"
- and critically: "**with less training data, attention sink disappears**" — the paper's
  own conclusion is that a data threshold matters more than model size

So the evidence is **contested, not settled**, and it points both ways depending on whether
you vary window size or training distribution. We killed F1 on one side of a live argument.
That does not resurrect F1 as novelty, but the premise we used to bury it is not safe.

## THE QUESTION THIS OPENS (best of the lap)

> **Does an attention sink form under a DISTILLATION objective, and where is its data
> threshold when the body is ~110K parameters?**

Every sink result above comes from next-token-prediction pretraining. Ours is KL to a
teacher distribution over a reduced vocabulary, from scratch, on 25-token sequences, with a
body three orders of magnitude below the smallest model in that literature.

Why this one is worth the time and the others weren't:

1. **It checks a SPECIFIC published claim** ("with less training data, attention sink
   disappears") rather than testing a truism.
2. **The runs already exist.** `ideaD-data*` is a trained sweep at 256 / 512 / 1024 / 2048
   / 4096 examples x 5 seeds. That is a data-threshold sweep sitting on disk.
3. **Attention is extractable with no framework change** — `torch.softmax(attention_scores)`
   at `transformer.py:348`.
4. Either outcome says something: sink present at 110K params under KL = the phenomenon is
   far more universal than the literature's floor suggests; sink absent = a concrete
   boundary condition on an ICLR claim.

## PITFALL — and it is a serious one

**Our models have no BOS token.** `student.py` encodes with `add_special_tokens=False`
everywhere. The canonical attention sink is *on BOS*. Without one, the sink either relocates
to the first content token or fails to form, and neither is directly comparable to the
literature. This must be established before any sink claim, and it may be the single thing
that decides whether the question is answerable at all.

Secondary pitfalls:
- **1 head x 3 layers = 3 attention maps.** Sink work talks about specialised "sink heads"
  selected from dozens. Our head statistics are thin to the point of anecdote.
- **25 tokens** is near the window size the paper says prevents sinks from forming.
- Short sequences + tiny vocab may make attention trivially uniform, in which case there is
  nothing to measure.

---

## CONTRADICTION 2 — emergence as a metric artifact, at our scale

**"Are Emergent Abilities of LLMs a Mirage?"** (2304.15004, NeurIPS 2023) argues emergence
comes from (a) nonlinear/discontinuous metrics and (b) **"too few test data to accurately
estimate the performance of smaller models."**

(b) is a stated limitation of the field's *practice*, and it is exactly what our setup
removes — three-minute convergence means arbitrarily many small models with many seeds and
large test splits.

Our own instrument already shows the (a) half at work: **task accuracy is exact-match
(discontinuous) while per-token accuracy is continuous, and they dissociate.** On math,
student token accuracy climbs 0.45 -> 0.63 -> 0.66 across epochs while task accuracy sits
flat at 0.21. That is the mirage mechanism in miniature, on our own logs.

Honest verdict: this is a **replication of the mirage argument in a new regime**, i.e. the
same species as every other survivor. It is a good thesis chapter and a good figure. It is
not a new mechanism.

---

## SCOPED CLAIMS worth mining (stated limits, not yet checked hostilely)

- Sink work is scoped to next-token pretraining. **Distillation objectives are unexamined.**
- Sink work's smallest from-scratch model is 60M; smallest observed is Pythia-14M. **Nothing
  below that.**
- Circuit-stability work ([2510.00845], [2602.22968]) reports feature reproducibility across
  **3 seeds** (21% at cos 0.7 for a 14.5M model) and an 8-seed distribution where a single
  seed landed at the top. **Everyone is seed-starved because retraining is expensive.** We
  have 10-seed cohorts and could afford 50.
- "Attention is not explanation" (Jain & Wallace 2019) vs Wiegreffe & Pinter 2019; Jacovi &
  Goldberg 2020 establish that **ablation-based** explanations satisfy faithfulness where
  correlational ones do not. Design consequence: measure "highest-yield" positions by
  **causal ablation**, never by raw attention magnitude.

---

## QUESTIONS RANKED BY (cheapness x chance of a non-obvious answer)

| # | Question | Cost | Runs exist? | Risk |
|---|----------|------|-------------|------|
| 1 | Does a sink form at all with no BOS, 25 tokens, KL objective? | Hours | Yes | May be trivially "no" -> dead end, but cheap and decisive |
| 2 | Where is the sink's data threshold on our `ideaD` sweep? | Hours | Yes (256-4096) | Depends entirely on Q1 |
| 3 | Cross-task attention divergence vs within-task SEED divergence (shared 3,840 sentences, reddit vs post-gen) | Hours | One size only | If seed noise >= task difference, the whole attention direction dies |
| 4 | Output-projection label geometry: ordinal structure, CE vs KL, across seeds | Free (weights only) | Yes | Teacher entropy 0.021 predicts a null |
| 5 | Discontinuous vs continuous metric divergence across the capacity sweep | Cheap | Partly | Replication of 2304.15004 |

**Run order: 1 and 4 first — both are hours-or-free and both can kill their branch outright.**
Q3's seed-noise control is the gate for the entire attention direction; if cross-task
divergence does not exceed within-task seed divergence, stop.

---

## HONEST OVERALL

Nothing in this lap is a new mechanism either. What is different from laps 1-4: items 1 and 2
check a **specific published claim** in a regime that claim never covered, on runs that
already exist, and a contradiction there would be a real result rather than a truism. That is
the highest ratio of "cheap" to "non-obvious answer" we have found.

The BOS pitfall may kill items 1-2 outright. Establish it first, in that order, today.
