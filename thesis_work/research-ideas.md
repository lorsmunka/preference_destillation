# Research Ideas — Domain-Specific LLM Distillation (TDK 2026 / folyóirat)

> **v2 — re-prioritized to your feedback.** Reordered by *your* enthusiasm, ideas 1+2 merged & demoted to last-resort, 7 and 9 shelved, and **3 new ideas added** (marked 🆕) that fit what you actually liked: the forecast/framework tool, deployable angle-wins, long-context, and optimization stacking. Novelty verdicts from the **24 prior agent-runs (2 passes)** are kept inline as protection for your related-work section. Jump to the **[TLDR table](#tldr)**.

---

## What changed and why (your steer)

Your decisive input: **on your runs, dark-knowledge gains are marginal on sharp/structured domains (CE ≈ KL).** That kills the *magnitude bet* — so any idea whose payoff requires "dark knowledge is a clear win" (old Ideas 1 & 2) drops to **last resort**, and the ideas that **turn your null result into the contribution** rise to the top:

- **Lead with the forecast/framework tool** (your "great") — its headline value is precisely telling people *when distillation is NOT worth it*, which your data already shows.
- **Pair it with a new finding** (🆕 P2): for sharp domains you can skip teacher *distributions* entirely and train on teacher *labels* (CE) — collapsing your 417-hour / 512 MB-per-batch bottleneck. Your "marginal dark knowledge" becomes a *positive, money-saving result*.
- **Promote the deployable angle-wins** you liked: the defer-token cascade (P3, "kinda works already") and the long-context tiny expert (P4, "I like idea 6").
- **Reframe optimization** (P5, 🆕) from "just another layer" into *the map of which optimizations compose vs interfere* — that's a finding, not a layer.

Two axes remain off-limits as *novelty* claims (cite, don't claim): **loss scheduling** (EGAD/AdaKD/…) and **efficient-attention distillation** (X-EcoMLA/TransMLA/NSA-R). Your edge is the **reduced-vocabulary narrow-expert regime** and the **practical framework** around it.

---

## P1 — Distillation pre-flight / ROI forecast tool ⭐ (your favorite; framework contribution)

A configurable analyzer that, before you spend a GPU-week, tells you whether to distill a domain, how, and how small.

- **Pitch:** "One cheap pass over the teacher → a go/no-go-and-how report: is KD worth it here, pure-CE or KL, min output vocab, min model size, data + storage cost."
- **Angle:** *I build* the missing pre-distillation planner — turning your scattered tooling into a framework whose flagship feature predicts distillation ROI per narrow domain from teacher-distribution statistics.
- **Your read:** "great… larger codebase refactor to make it truly framework-like, but that's okay." → This is the framework-ification your TDK proposal already wants ("jó alapbeállításokkal rendelkező framework"). The refactor *is* part of the contribution.
- **Verdict (2 passes): SURVIVES but squeezed** — the KD-vs-CE leg is owned by **Menon 2021** (variance-reduction ∝ teacher distance from one-hot); min-size is squeezed by scaling laws (2502.08606) and a minimum-width theorem (2604.04037, but needs expensive SAEs). **Unclaimed = the cheap, joint, per-domain instrument** across the sharpness spectrum. So sell it as *tooling/operationalization*, not a new mechanism.
- **Why it wins:** Practitioner-shaped, demo-able, and your 100+ runs are the evidence base. A framework + a decision tool reads as mature engineering research, which TDK juries reward.
- **Pipeline effort: MED–HIGH** (the refactor). Entropy stats are trivial from saved logits; min-size data largely exists (`evaluate_runs.py`). Bake in an **auto-recipe default**: read teacher sharpness → pick CE-only vs KL automatically (the "good default" your proposal calls for).
- **Decisive output → metric:** a one-page report per domain (entropy → predicted CE-vs-KL gap, min vocab [P7], min params, GB of distillation data) validated against held-out domains; report prediction error.
- **Cite (up front):** Menon ICML 2021; Distillation Scaling Laws (2502.08606); Teacher Calibration (2508.20224); GRACE (2511.02833); minimum-width (2604.04037).
- **Risk:** Low technical risk; the refactor is the cost. **Tier 1.**

## P2 — 🆕 "Do you even need the distributions?" — teacher-label CE vs teacher-distribution KL

The honest, money-saving finding hiding in your null result: for sharp domains, training on teacher *labels* (CE) matches full distribution distillation — so skip the expensive logit extraction.

- **Pitch:** "On structured domains you don't need the teacher's distribution at all — argmax labels + reduced vocab get you the same student, at a fraction of the data-gen cost and storage."
- **Angle:** *I quantify* the cost/information tradeoff of distillation data across the sharpness spectrum — how many accuracy points per stored byte the teacher's distribution actually buys over its labels — and show the break-even domain where distributions start to matter.
- **Verdict / novelty (honest, not yet hostile-verified):** The *principle* (soft labels ≥ hard labels in sample efficiency) is **Hinton-owned** — so this is a **practical/empirical** contribution, **low novelty as a claim** but high practical value and directly feeds P1. Differentiator: a **per-domain info-per-byte curve + a "skip the distributions" decision rule** tied to teacher sharpness; the storage payoff (argmax/tiny-top-k for sharp, full for soft) connects to (but is more decision-driven than) Sparse Logit Sampling (2503.16870). Frame as "when the expensive part is unnecessary," cite Hinton/Menon.
- **Why it wins:** It's a **contrarian, useful** result that *uses your existing runs* and slays your worst bottleneck (data-gen = 417 h; post-gen batches = 512 MB). Reviewers love "we show you can do less."
- **Pipeline effort: LOW.** You already ran pure-CE vs pure-KL vs annealed on sentiment + math (Table 2 in the thesis already hints CE≈KL). Add the *cost* axis (bytes, gen-time) and the soft-domain contrast.
- **Decisive experiment → metric:** CE-on-labels vs KL-on-distributions task accuracy **and** data cost (GB, gen-hours), per domain → plot **accuracy-gain-per-GB vs teacher sharpness**; mark the break-even.
- **Cite:** Hinton (1503.02531); Menon ICML 2021; Sparse Logit Sampling (2503.16870); your own Table 2.
- **Risk:** Low. **Tier 1** (keystone pairing with P1). Note: this partly *replaces* the dark-knowledge-magnitude bet you distrust — instead of proving dark knowledge helps, you profit from showing when it doesn't.

## P3 — Tiny-expert cascade with a trained defer-token ⭐ (you: "cool… kinda works already")

The closed vocabulary makes off-domain inputs self-detecting; the tiny model routes to the teacher. Now turned into a measured end-to-end system.

- **Pitch:** "A sub-10M on-device expert that knows what it doesn't know, defers to the teacher when unsure, and the blended system is cheaper at equal accuracy."
- **Angle:** *I came up with* a dedicated **`defer` sink token** trained into the closed vocabulary via the KL loss (teacher off-vocabulary mass → the token), giving a one-number cascade trigger — then I measure the **end-to-end cost/accuracy/latency** of the tiny-first→teacher system.
- **Your read:** "cool… basically an angle change, it kinda works already." → Exactly: low effort, high tangibility, and the cascade-system framing upgrades it from "angle" to "deployed system with numbers."
- **Verdict (2 passes): SURVIVES (clean).** Deferral mechanism owned (MSP, cascade deferral, Gatekeeper); closest threat **Expert-Token-Routing (2403.16854)** routes among peer experts on a full-vocab model with no defer/OOD semantics. **Unclaimed:** the **vocabulary reduction *causes* the signal**, and the defer prob is **learned via KL mass-routing from the teacher**.
- **Pipeline effort: LOW–MED.** You have reduced in/out vocab + `remap_input_tokens`. Add the reserved defer token, a cheap input-OOV gate (single pass), teacher-labeled off-domain examples, and a blended-cost harness.
- **Decisive experiment → metric:** **AUROC** of defer-token mass separating in/out-of-domain; **cost-vs-accuracy** of the cascade vs (a) tiny-only, (b) teacher-only, (c) MSP baseline.
- **Cite:** Expert-Token-Routing (2403.16854); Gatekeeper (2502.19335); MSP (1610.02136); token-level deferral CITER/R2R (2404.10136). ⚠️ cite "Entropy Alone is Insufficient…" → lean on the trained token, not raw entropy.
- **Risk:** Low. **Tier 1 — best demo.**

## P4 — Long-context tiny on-device expert via footprint-aware attention (you: "I like idea 6")

Distill an older open teacher into a tiny reduced-vocab student with cheap attention for long-context narrow tasks. De-scoped to be finishable.

- **Pitch:** "A sub-10M reduced-vocab expert that reads 16k-token documents on-device, because windowed attention + a tiny output space keep the footprint in budget."
- **Angle:** *I build* a recipe + Pareto curve for tiny domain-vocabulary long-context experts, swapping the student's attention (full → sliding-window+GQA → MLA-lite) at fixed parameter budget.
- **Verdict (2 passes): generic mechanism SCOOPED; salvageable as a systems/empirical PoC.** ⚠️ **Two corrections the jury will make:** (1) "tiny vocab → cheaper attention" is **mechanistically false** (vocab isn't in the KV-cache formula) — the honest link is *parameter-budget reallocation*; (2) distilling into MLA/NSA is done (X-EcoMLA, Foreign Sparse Attention). So claim a **recipe + Pareto**, not a mechanism.
- **De-scoped GO plan:** **don't** build MLA/NSA/DSA from scratch (kernel pain; pays only at 64k). Use **sliding-window + GQA** (~1 day; NSA's local branch; what Gemma ships); **MLA-lite** (single low-rank KV) optional to name-drop DeepSeek honestly. **New domain:** long transcript/doc → deterministic JSON (keeps output vocab tiny; precedent DiDOTS 2410.04188).
- **Pipeline effort: MED** (the new long-context domain + data-gen is the real cost; attention swap is cheap).
- **Decisive experiment → metric:** 3 students differing *only* in attention; sweep context 1k→16k; plot **task-JSON accuracy + peak KV memory** → **memory-vs-accuracy Pareto at fixed params**. Pick docs >4k tokens or there's no result.
- **Cite:** TransMLA (2502.07864); X-EcoMLA (2503.11132); MobileLLM (2402.14905); DeepSeek-V3.2/DSA (2512.02556); DiDOTS (2410.04188).
- **Risk:** Med effort, low mechanism-novelty — sell as "honest finishable recipe for on-device long-context narrow experts," topical via DeepSeek. **Tier 2.**

## P5 — 🆕 Narrow-expert optimization stack: how small can it get, and what composes

Stack every cheap optimization and map which ones compose vs interfere — the umbrella that absorbs the quantization idea you called "another layer."

- **Pitch:** "Input-vocab cut × output-vocab cut × quantization × windowed attention — how small can a narrow expert really get, and which of these step on each other?"
- **Angle:** *I measure* the composed footprint frontier of a narrow distilled expert and the **interaction map** — e.g., does int4 hurt *more* after vocab reduction? does windowed attention interact with a tiny vocab? — which "stack the tricks" papers never test.
- **Your read:** "okay. just another optimization layer?" → Right, individually each is a layer; the **research is the interaction map** (what composes multiplicatively vs interferes), not any single layer.
- **Verdict / novelty:** Each lever is known and "compression techniques compose" is established; the **closest neighbor for the quant leg is FlashHead (2603.14591)** (head quant via fewer effective classes). **Unclaimed = the controlled vocab×bitwidth×attention interaction study** with the key contrast *head-quantized vs head-FP16* (does a tiny vocab let you drop the near-universal FP16-lm_head carve-out?).
- **Why it wins:** Pure-tangible, on-device, "look how small" — and the interference finding gives it a real result even if individual layers are known. Sub-MB experts demo well.
- **Pipeline effort: LOW–MED.** Mostly PTQ + ablations on existing checkpoints (a weekend for the core); on-hardware energy is the only time-sink (measure memory/latency, skip energy unless needed).
- **Decisive experiment → metric:** 2-D+ sweep **vocab {525, mid, full} × bits {fp16,int8,int4} × attn {full, window}**; metric = **degradation slope vs FP16 footprint**, key contrast **head-quant vs head-FP16**. Crossover (tiny-vocab tolerates a quantized head where full-vocab collapses) = the result.
- **Cite:** FlashHead (2603.14591); VocabTailor (2508.15229); VQ-Logits (2505.10202); "Give Me BF16 or Give Me Death" (2411.02355); compression-composability surveys.
- **Risk:** Low effort, outcome-dependent on the interaction being non-trivial — **probe before committing**. **Tier 2/3.**

## P6 — Reduced output vocabulary as a distillation accelerator (you: "okay-ish")

At matched body, the output cut speeds convergence in *examples* — and the closest paper argues the opposite, so it's contrarian.

- **Pitch:** "Truncating the output head concentrates the KL gradient on live classes → fewer examples to converge, not just fewer params."
- **Angle:** *I prove* a sample-efficiency (examples-to-target) gain from a fixed reduced output vocab in distillation at matched capacity/data.
- **Verdict (2 passes): SURVIVES (clean, contrarian).** The closest threat — **"The LM Head is a Gradient Bottleneck" (2603.10145)** — argues a *fuller* head helps gradient flow, so your "shrinking helps" is contested, not anticipated.
- **Pipeline effort: LOW–MED.** Decisive ablation = one-line swap (full 262k vs reduced 525 head, same body) + the **mask-dead-classes vs shrink-head** control that isolates gradient-concentration from parameterization.
- **Decisive experiment → metric:** **examples-to-target-KL/-accuracy** (NOT wall-clock); arms full / reduced / full+top-k-mask.
- **Cite:** Hinton (1503.02531); LM-Head Gradient Bottleneck (2603.10145); Extreme Classification (2002.06298); Adaptive Sparse Softmax (2508.03175).
- **Risk:** Low. **Tier 2** (kept middle per your read).

## P7 — (Last resort) Output-vocabulary design: budget + token selection [merged old Ideas 1+2]

How small the output vocab can go, and which auxiliary tokens to keep — the dark-knowledge bet you (rightly) distrust on sharp domains.

- **Your read:** "1 and 2 are last resorts… they converge." → Correct: both rest on the *same bet* — that the output distribution's dark knowledge is worth optimizing — which your runs show is **marginal on sharp domains**. So this is a **fallback**, best aimed at **soft domains only** (post-generation), or used as *evidence* feeding P1/P2 rather than a headline.
- **Pitch:** "If dark knowledge ever pays (free-form domains), here's the measured floor on output-vocab size and the token-selection policy that captures it."
- **Verdict (2 passes):** *Budget* (old Idea 1) = **SURVIVES clean** (measured generative threshold curve vs entropy is unclaimed; principle owned by Subclass Distillation). *Selection* (old Idea 2) = **SURVIVES but incremental** (rests on a horse-race that may come out flat; per-token dynamic selection is owned by SLIM/Sparse Logit Sampling).
- **Why it's last-resort for you:** the effect size is the risk, and your runs already whisper "small." Only worth running on the **free-form/post-gen domain**, where dark knowledge is real — and there it doubles as evidence for P1/P2's "soft domains need distributions" claim.
- **Pipeline effort: LOW** (offline from saved logits). **Decisive metric:** KL-component & task accuracy vs output-vocab size, per domain, overlaid on teacher entropy; selection policies at fixed budget — **report the structured→free-form regime curve** (where the gap *grows*).
- **Cite:** Subclass Distillation (2002.03936); Hinton (1503.02531); Rethinking Selective KD (2602.01395); Kolesnikova (2205.02340).
- **Risk:** Medium (effect may be inside noise on sharp domains — which is itself P2's finding). **Tier 3 / fallback.**

---

## Shelved (your call)
- **Safety/robustness by construction** (old Idea 7) — "good but not my cup for now." Parked. (Only viable as a measured CDA-immunity security study anyway; 2503.24191.)
- **Entropy-budgeted dynamic vocabulary** (old Idea 9) — "meh." Dropped from the active set. (Was the highest-novelty *method*, if you ever want a prestige angle; resurfaces naturally from P7.)

---

## Recommended thesis spine (re-tuned to your taste)
1. **P1 (forecast/framework tool)** as the backbone + the refactor → your "framework with good defaults" goal.
2. **P2 (do you need the distributions?)** as the headline empirical finding feeding P1 — turns your CE≈KL reality into a money-saving result.
3. **P3 (defer-token cascade)** as the deployable demo.
4. **P4 (long-context PoC)** as the ambitious, topical extension (new domain).
5. **P5/P6** as optimization/efficiency chapters; **P7** only on the free-form domain or as P1/P2 evidence.

**Honesty note:** P1/P2/P5 are practical/empirical (low novelty-stakes, high usefulness) — perfect for a framework thesis; P3/P6 carry the cleaner *novel* slivers; P4 is a systems PoC. None bets on dark knowledge being a clear win — which is exactly what your runs told us to avoid.

**Not yet hostile-verified:** P2 and P5 are new this round; I gave honest preliminary novelty reads but did **not** run the kill-the-sliver agents on them. Say the word and I'll red-team both before you commit.

---

## <a id="tldr"></a>TLDR — priority order (your steer + novelty)

| P | Idea | One-line | Your read | Novelty | Effort | Tier |
|---|------|----------|-----------|---------|--------|------|
| 1 | **Forecast / ROI framework tool** | cheap teacher pass → distill? how? how small? | "great, refactor ok" | Survives, squeezed → sell as tooling | Med–High | **1** |
| 2 | 🆕 **Do you need the distributions?** | CE-on-labels ≈ KL on sharp domains → skip logit extraction | (new, from your CE≈KL runs) | Low-novelty principle, high practical value | Low | **1** |
| 3 | **Defer-token cascade** | closed vocab → self-aware OOD → tiny-first system | "cool, kinda works already" | Survives (clean) | Low–Med | **1 demo** |
| 4 | **Long-context tiny expert** | reduced-vocab + windowed attn, long docs→JSON | "I like idea 6" | Mechanism scooped → PoC | Med | 2 |
| 5 | 🆕 **Optimization stack** | input×output vocab × quant × attn — what composes? | "okay, another layer?" → it's the interaction map | Low; FlashHead neighbor | Low–Med | 2/3 |
| 6 | **Vocab as accelerator** | reduced head → fewer examples to converge | "okay-ish" | Survives (clean, contrarian) | Low–Med | 2 |
| 7 | **Output-vocab design (budget+selection)** | min vocab + which aux tokens [merged 1+2] | "last resort; they converge" | Budget clean / selection incremental | Low | 3 / fallback |
| — | ~~Safety by construction~~ | harmful tokens absent | "shelf it" | — | — | shelved |
| — | ~~Dynamic vocabulary~~ | per-token entropy-budgeted output | "meh" | (highest method-novelty) | — | shelved |

**Provenance:** verified in code (`distillation_data_generation/model_handler.py` saves per-step teacher logits over the reduced vocab; `training/trainer.py:_compute_loss` = global cosine KL/CE + temperature). Novelty: 24 agent-runs / 2 passes; all 5 originals survived a hostile examiner. v2 re-prioritized to your feedback (2026-06-01): dark-knowledge-magnitude bets (old 1,2) → last resort; forecast tool + "skip the distributions" + deployable systems → front; safety & dynamic-vocab shelved; +2 new ideas (P2, P5).
