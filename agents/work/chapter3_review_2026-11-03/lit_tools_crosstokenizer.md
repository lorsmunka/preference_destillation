# Literature check: toolkits, cross-tokenizer KD, sparse/top-k storage, KL vs CE, annealing

Checked 2026-11-03 for thesis section 3.1.6 (and 2.x where it touches the loss and the vocabulary).
Scope: toolkits, cross-tokenizer / vocabulary mismatch, vocabulary trimming, sparse logit storage, KL vs CE, ratio/temperature annealing.
Not covered: small LMs and task-specific KD in general (the other agent's part).

Legend: every line marked **[V]** was checked on the primary page (arXiv abstract or HTML, OpenReview/ACL Anthology, official docs, GitHub README or source, official blog). **[UNVERIFIED]** = not opened, or the page did not state it. "inference" = my reading, not stated on the page.

Already verified 2026-11-01 (not redone): GKD (arXiv 2306.13649, ICLR 2024), MiniLLM (2306.08543), DistillKit basics, torchtune KD recipe (fixed kd_ratio), Minitron (2407.14679), EasyDistill (2505.20888). Changes since then are in section 6.

---

## 1. Cross-tokenizer distillation

The thesis case: the student's output vocabulary is a **fixed subset of the teacher's own tokens** (ids map 1:1), and the teacher's logits are renormalised over that subset offline. So there is **no alignment problem**: there are no sequence-alignment issues (the same tokenizer segments the text) and no token-to-token mapping (each student output = one teacher token). The only loss is the teacher mass outside the subset (section 3). Cross-tokenizer methods solve a harder problem: two **different** tokenizers, different segmentations, partly overlapping vocabularies, and always a **pretrained** student with its own full vocabulary.

| Method | Source [V] | Mechanism | Teacher | Student (size, init) | Vocab relation |
|---|---|---|---|---|---|
| ULD | Boizard et al., arXiv 2402.12030, TMLR Jan 2025 | Optimal-transport (Wasserstein) loss. All costs set to 1, so it becomes the L1 distance between **sorted** probability vectors; the smaller vocabulary is zero-padded. Loss = CE + λ·ULD, λ = 1.5 | Llama 2 7B Chat, Mistral 7B Instruct | OPT-350m, Pythia-160m/410m/1b, Bloomz-560m, mT0-580m; **pretrained** | Different tokenizers |
| DSKD | Zhang et al., arXiv 2406.17328, EMNLP 2024 | Projects hidden states into the other model's space and uses one shared prediction head. Cross-model attention aligns tokens across vocabularies | GPT2-1.5B / Qwen1.5-1.8B; LLaMA2-7B / Mistral-7B | GPT2-120M, TinyLLaMA-1.1B (pretrained checkpoints; implied, the page does not say "pretrained") | Same and different vocabularies both tested; compares KL, RKL, JS, SKL, SRKL, AKL |
| DSKD v2 (+ ETA) | Zhang et al., arXiv 2504.11426 (Apr 2025, "under review") | The same dual-space idea, plus an exact token alignment algorithm; off- and on-policy | — | — | Different vocabularies |
| MinED (FuseLLM) | Wan et al., arXiv 2401.10491, ICLR 2024 | Token alignment across tokenizers by **minimum edit distance** (relaxes exact match; e.g. aligns "gets" with "get") | Llama-2 / OpenLLaMA / MPT 7B (fusion sources) | Llama-2 7B target | Different tokenizers |
| Multi-Level OT | Cui et al., arXiv 2412.14528, AAAI 2025 (oral) | Token-level and sequence-level OT (Sinkhorn); "no dimensional or token-by-token correspondence" needed; CE + the OT losses | LLaMA2-7B-Chat, Mistral-7B-Instruct, Qwen-7B-Chat, LLaMA3-8B-Instruct | OPT-350M, Pythia-160M/410M/1B, Bloomz-560M, mT0-300M; "initializing them with their pretrained weights" | Different tokenizers |
| CDM | Chen et al., arXiv 2502.11104, Findings of ACL 2025 (pp. 8005–8018) | Entropy-weighted DTW sequence alignment + dynamic contextual top-K vocabulary mapping | 5 families (Llama3, Phi3, Gemma2, OPT, Qwen2), 3 pairs | sizes not on the abstract page [UNVERIFIED: Qwen2-0.5B from a search snippet] | Different tokenizers |
| ALM | Minixhofer, Vulić, Ponti, arXiv 2503.20083, NeurIPS 2025 | Approximate likelihood matching over comparable token chunks (binarised f-divergence); "pure distillation", not an auxiliary loss | OpenMath2-Llama3.1-8B (math case) | Gemma2 2B (an existing released model; pretrained presumably, not stated) | Fundamentally different tokenizers, incl. subword to byte |
| X-Token | Turuvekere Sreenivas et al. (NVIDIA), arXiv 2605.21699, 20 May 2026 | Sparse projection matrix W from student to teacher vocabulary, built from tokenizer string rules; P-KL and H-KL losses; multi-teacher | Qwen3-4B, Phi-4-Mini, Llama-3B | Llama-3.2-1B (pretrained presumably; the abstract does not say) | Different tokenizers. Reports +3.82 avg points over GOLD (Qwen3-4B teacher) |
| GOLD (TRL) | HF blog "Unlocking On-Policy Distillation for Any Model Family", Patiño et al., 29 Oct 2025 (update June 2026); TRL docs `trl.experimental.gold` | Extends ULD to **on-policy**: aligns text spans of both tokenizers, merges the probabilities of split tokens (chain rule), hybrid loss (JSD on exactly matched tokens, sorted-L1 ULD on unmatched ones) | Qwen2.5-7B-Instruct, Qwen3-4B-Instruct-2507 | Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct, gemma-3-1b-it (**pretrained instruct models**) | Different tokenizers; token-content similarity Llama/Qwen 0.64, Gemma 0.063; performance falls as similarity falls |
| On-Policy Distillation (Thinking Machines) | Kevin Lu et al., thinkingmachines.ai blog, 27 Oct 2025 | Per-token **reverse KL** on student-sampled tokens, used as the advantage in an RL update; discount 0; runs on Tinker | Qwen3-32B (math; Qwen3-8B also tried) | Qwen3-8B-Base after SFT mid-training (not raw); June 2026 cookbook: Qwen3.5-9B-Base | Tokenizer requirement **not stated** [UNVERIFIED]; inference: the teacher scores the student's own token ids, so a shared tokenizer is implied |

GOLD naming: TRL docs say "General **Online** Logit Distillation"; the HF blog says "General **On-Policy** Logit Distillation". Both [V]. Use the blog's name with the TRL path.

Key numbers [V], HF blog (Countdown task): GOLD with Llama-3.2-1B ← Qwen3-4B raised the student by 25% and recovered 60% of the teacher's performance, against +5% / 10% for ULD. Same-tokenizer GKD recovered over 80%. Cross-tokenizer GOLD was about 20% better than GRPO.
Thinking Machines [V]: AIME'24 went from 60% to 70% in about 150 steps. The cited Qwen3 report gives RL 67.6% for 17,920 GPU-hours against on-policy distillation 74.4% for 1,800 GPU-hours. The blog estimates a 9–30x FLOP reduction against SFT.

**How the thesis differs (fact list):**
- Subset, not a different tokenizer: student output id k ↔ one teacher token id. No span merging, no OT/sorting, no projection matrix, no edit-distance mapping.
- The student input side also uses the teacher's tokenization (a compact input vocabulary plus an unknown-token row). Segmentation is identical, so no sequence alignment is needed.
- All cross-tokenizer works above use **pretrained** students of 120M–9B. None trains a student from scratch.
- The nearest "subset" mechanisms in tools restrict only the **loss support**, while the student keeps its full head (see 3 and 6): ms-swift top-K renormalisation, X-Token's `xtoken_vocab_topk` (global top-k teacher tokens), NeMo RL P-KL "microbatch-global top-k subset".

---

## 2. Vocabulary trimming / reduction

| Work | Source [V] | What is reduced | How chosen | Numbers | Relation to the thesis |
|---|---|---|---|---|---|
| Vocabulary Trimming (VT) | Ushio, Zhou, Camacho-Collados, arXiv 2305.15020, Findings of EMNLP 2023 | Input+output vocabulary of a multilingual LM, to one language | Keep tokens that appear in a language corpus (mC4), or the top-n most frequent | mT5-small 250K vocab / 300M params; Korean 73K (29.2%) / 119M (39.7%); top-5K ≈ 49M params (16% of mT5); "around 50%" of the vocabulary is enough; performance kept with <40% of the parameters. Pre-FT and post-FT VT | **Subset of the original vocabulary**, as in the thesis. But it is pruning of a pretrained model, not distillation, and the subset comes from a language, not a task |
| VocabTrim | Goel et al. (Qualcomm), arXiv 2506.22694, ICML 2025 workshop (ES-FoMo) | Drafter LM head in speculative decoding | Tokens "most frequently sampled from the vocabulary of the target model"; training-free | +16% memory-bound speed-up, Llama-3.2-3B-Instruct, Spec-Bench | Subset of the target's tokens chosen from the target's own outputs: **closest in spirit** to the thesis's task-derived vocabulary, but for a drafter, training-free, and with no KD |
| FR-Spec | Zhao et al., arXiv 2502.14856 (Feb 2025) | Draft vocabulary | Frequency-ranked subset | −75% LM-head compute, 1.12x over EAGLE-2; output distribution unchanged (verification by the full model) | As above |
| Mixed-vocabulary training | Zhao, Gupta, Song, Zhou, arXiv 1909.11687, EACL 2021 | Student BERT vocabulary 30,522 → 4,928 WordPieces | **Separately trained** WordPiece on the same corpus. The teacher vocabulary "includes 93.9% of the WP tokens in this 5K student vocabulary but does not subsume it" | Students 6.2M / 10.9M (GLUE), 1.2M–6.2M (SNIPS) | Tiny reduced-vocabulary students exist (encoder, MLM). Not a strict subset, no logit KD (the method aligns embeddings; Stage II is student-only MLM) |
| KD with vocabulary reduction (Russian) | Kolesnikova et al., arXiv 2205.02340 (2022, preprint) | Student vocabulary 120K → 5K–30K | Separately built BPE; ~99% of the subwords also in the teacher's | 17x–49x compression; "Match" = KD only on shared subwords (loses 75–96% of usable subwords); "Reduce" = sum student pieces per teacher subword; students **initialised from the teacher** | Encoder (BERT-style; inference), not from scratch |
| TinyStories | Eldan & Li, arXiv 2305.07759 | Tokenizer | "We use GPT-Neo tokenizer but only keep the top 10K most common tokens" | Models ~1M–35M, from scratch | A subset of an existing tokenizer for tiny from-scratch LMs; the data is LLM-generated, but there is no logit KD |

Takeaway: restricting a model's output vocabulary to a frequency- or domain-derived **subset of an existing vocabulary** is documented (VT, VocabTrim, FR-Spec, TinyStories). Reduced-vocabulary **students** under KD are documented for BERT encoders (Zhao 2019/2021, Kolesnikova 2022). Not found: a reduced vocabulary derived from the **task's teacher outputs**, used as the **only output head of a from-scratch autoregressive student** with **logit KD over exactly that subset**.

---

## 3. Sparse / top-k logit storage for offline KD, and the bias question

| Work | Source [V] | Finding |
|---|---|---|
| Sparse Logit Sampling | Anshumann et al., arXiv 2503.16870, **ACL 2025 (oral)** | **Yes**: caching Top-K probabilities "provide[s] biased estimates of teacher probability distribution to the student, resulting in suboptimal performance and calibration". Mechanism (HTML): with or without renormalising the Top-K, the non-Top-K tokens get target 0. The gradient becomes (Σ_{j∈K} t_j)·p_i − t_i, so the student learns an **up-scaled teacher**: "over-confident in the Top-K tokens, and under-confident for the remaining tokens". ECE gets worse as K decreases. Top-K is L1-optimal per token (L1 = 2(1−a), a = retained mass) but still biased. Fix: Random-Sampling KD (importance sampling, unbiased in expectation, ~12 unique tokens, ~3 bytes each; overhead <10% vs CE). Setting: **pretraining** distillation, students 100M–3B, teachers 3B/8B (33B for the 300M runs), K ∈ {1…300} |
| Gemma 3 tech report | arXiv 2503.19786, §2.2 | Gemma 3 itself was distilled with "256 logits per token, weighted by teacher probabilities"; non-sampled logits get "zero probability … and renormalized"; CE loss. Renormalised sparse targets are mainstream practice, used for the thesis's own teacher family |
| ms-swift GKD Top-K mode | swift.readthedocs.io v4.0 GKD page | Top-K indices "selected from the teacher model". Both teacher and student probabilities are "renormalized over the Top-K subset" before the JSD. Required with an external teacher server (vLLM `max_logprobs` default 20, per the v4.3 docs [UNVERIFIED: only the v4.0 page opened]) |
| DistillKit sparse KL | GitHub source `distillkit/lossfuncs/kl.py`, `missing_probability.py` | Option `missing_probability_handling`: `ZERO` (assume the missing mass is 0) or `SYMMETRIC_UNIFORM` (spread the missing mass uniformly over the missing tokens for teacher and student, adding a missing-mass KL term). So the tail issue is handled explicitly in code (not in the README) |
| Pre-training Distillation design space | Peng et al., arXiv 2410.16215 (Oct 2024) | Top-p 0.95 + top-k 100 cut storage ~4,000x (≈15 TB / 100B tokens); with p = 0.95, k = 50 was best; **every k tested beat LM-only, including k = 1** |
| Don't Ignore the Tail (TAD) | Dasgupta, Cohn, Baldwin, arXiv 2602.20816, ICML 2026 | Standard KL is "dominated by" the highest-probability tokens; a tail-aware divergence decouples the top-K mass from the rest |
| BiLD | Li, Zhou, Song, arXiv 2406.13555, COLING 2025 | Argues the opposite for task-specific KD: the long tail is **noise**; top-8 logits beat SFT, vanilla KL and 5 other methods on 13 datasets. ">99% mass in top-1024" is [UNVERIFIED] (not on the abstract page) |
| Offline Top-K + fused chunked KL | Ryskulov et al., arXiv 2608.03796 (Aug 2026, preprint) | Offline top-K caching matches online at near-identical loss; ~29% faster per iteration. Tail handling not stated on the abstract page [UNVERIFIED] |

**Does this imply a bias the thesis should acknowledge? Yes, in a narrower form. Fact list:**
- The thesis target at each position = P_T(v) / m for v ∈ V, where m = the teacher's retained mass inside V at that position. Relative to the **full** teacher, this is the same up-scaling Anshumann et al. describe, by a factor 1/m per position.
- One half of their argument does **not** transfer: tail tokens pushed to zero although the student can emit them. In the thesis the student's head **is** V, so there are no "remaining tokens" to under-weight.
- Within V the renormalised target is the best the student can do. Among distributions supported on V, P_T(·)/m minimises both forward KL and cross-entropy to the teacher (standard result: maximising Σ_{v∈V} P_T(v) log Q(v) subject to Σ_V Q = 1 gives Q ∝ P_T). So the bias is "conditioned on V", not a gradient-estimator bias like per-position top-K.
- The bias is position-dependent. Where m is low (the teacher wants a token outside V), the target is sharper than the teacher → overconfidence at exactly those positions. Teacher-entropy figures, top-k overlap and calibration comparisons are then **relative to the renormalised teacher**, not the real one.
- Difference from top-K caching: the subset is fixed per task, dense over V (bf16, every token in V stored at every position), and the same at every position. There is no per-position truncation on top of V.
- What to report: the distribution of m (mean, a low percentile, min) for each domain's teacher. The thesis currently cites only a Gemma 3 note (see the memory reminder "retained mass": re-measure on Gemma 4 for both agents). If mean m ≈ 1 the effect is negligible; say so with the number.
- Check item: in teacher-**generated** mode, how are teacher-sampled tokens outside V handled (were they possible at all)? Also, gold targets in teacher-forced mode must lie in V. State how this is guaranteed.

---

## 4. KL vs CE, forward vs reverse KL

| Work | Source [V] | Relevant fact |
|---|---|---|
| Hinton, Vinyals, Dean 2015 | arXiv 1503.02531 | Weighted average of the soft-target objective (high T) and the hard-label objective (T = 1); "best results … considerably lower weight on the second". Multiply the soft term by T². Soft targets with 3% of the data: 57.0% vs 44.5% for hard targets (58.9% with all the data) |
| Kim & Rush 2016 | arXiv 1606.07947, EMNLP 2016 | Word-level KD: L = (1−α)·NLL + α·KD, α = 0.5 better than 0.9. Small students (2x500 ≈ 84M, 2x300 ≈ 49M LSTM) vs a 221M teacher: word-KD over NLL baseline = +0.7 / +0.8 BLEU greedy, +0.1 / +0.7 beam. Small but consistent. (Students trained from scratch: standard for 2016 NMT, not explicitly stated [UNVERIFIED]) |
| GKD | arXiv 2306.13649 (HTML) | Compares SFT, supervised KD, SeqKD, on-policy GKD on pretrained T5 (77M/250M/800M) students; on-policy GKD best; relative gains 2.1x / 1.7x / 1.9x over baseline KD methods |
| Gemma 2 | arXiv 2408.00118, Table 6 | 2B trained on 500B tokens: **from scratch 60.3 vs distilled from 7B 67.7** (3-benchmark average). Pretraining-scale evidence that KD beats NTP/CE for a from-scratch student |
| Pre-training distillation | Peng et al., arXiv 2410.16215 | 1.9B student from scratch, GLM-4-9B teacher: pure KD 38.3 vs LM-only 37.7; static α = 0.9 best (+3.6%); best schedule +8.0% (section 5) |
| Distillation Scaling Laws | Busbridge et al., arXiv 2502.08606, ICML 2025 | Distillation beats supervised learning "up to a compute level that scales predictably with student size", when a teacher already exists; if the teacher must be trained for one student, supervised is "generally preferable". Students 143M–12.6B; pure distillation λ = 1 is statistically similar to the optimal λ*; τ = 1 best |
| Rethinking KL in KD for LLMs | Wu et al., arXiv 2404.02657, COLING 2025 | "neither mode-seeking nor mean-seeking properties manifest in KD for LLMs"; FKL and RKL "share the same optimization objective" and converge after enough epochs. Early on, FKL fits the head and RKL the tail; proposes AKL (adaptive mix) |
| DistiLLM | Ko et al., arXiv 2402.03898, ICML 2024 | Skew KL + adaptive off-policy use of student outputs; up to 4.3x speed-up over recent KD methods |
| Does KD really work? | Stanton et al., arXiv 2106.05945, NeurIPS 2021 | Student–teacher **fidelity** is often low ("a surprisingly large discrepancy"), mainly from optimisation; better fidelity does not always mean better generalisation |
| KD as label smoothing | Yuan et al., arXiv 1909.11723, CVPR 2020 | "KD is a type of learned label smoothing regularization"; even weak teachers help. An alternative explanation of KL > CE (regularisation, not "dark knowledge") |

**Framing the thesis result** (16M from-scratch math-tutor student, 3 seeds: final task accuracy KL .705 vs CE .683, KL ahead in 30/30 paired tests; post-generation top-20 overlap .305 vs .264):
- The direction is **expected and well documented**: Hinton 2015 (strongest in low data), Kim & Rush 2016 (small, consistent gains for 49–84M students), Gemma 2 (from scratch 60.3 → 67.7), Peng 2024, Busbridge 2025 (KD wins at small student / limited compute when the teacher exists).
- The magnitude (+2.2 points) matches the small but consistent gains in Kim & Rush.
- Higher top-k overlap for the KL student is near-trivial: KL optimises the match directly and CE never sees the teacher distribution. The low absolute level (~0.3) fits Stanton et al.'s fidelity gap.
- What is a new data point (not a new finding): the same direction for a **tiny (16M) from-scratch, single-task autoregressive student with a task-derived output vocabulary**, measured paired over seeds and evaluation passes. No reviewed paper reports KL vs CE in this exact setting (the closest: Kim & Rush, NMT 49–84M; Zhao 2019, tiny BERT but no logit KD).
- Possible confound to state: KL gets more information per token (a full distribution over V) as well as regularisation (Yuan 2020). The thesis cannot separate the two without e.g. a label-smoothing arm (a suggestion, not a requirement).

---

## 5. KL/CE ratio annealing and temperature schedules

| Who | Source [V] | What is scheduled |
|---|---|---|
| BAM! teacher annealing | Clark et al., arXiv 1907.04829, ACL 2019 | Target = λ·y_gold + (1−λ)·teacher; "λ is linearly increased from 0 to 1 throughout training". So the **KD share goes linearly 1 → 0**, the same direction as the thesis's KL ratio 0.9 → 0.1 and its kl-to-ce arm (1 → 0) |
| Annealing-KD | Jafari et al., EACL 2021, pp. 2493–2504 (arXiv 2104.07163) | Stage I: MSE between student logits and teacher logits × Φ(T), Φ(T) = 1 − (T−1)/τ_max, T stepped **down** from τ_max to 1 (k passes per T). Stage II: CE on hard labels at T = 1. Temperature annealing plus a hard switch to CE |
| CTKD | Li et al., arXiv 2211.16231, AAAI 2023 | Learnable, "dynamic" temperature with an easy-to-hard curriculum (adversarial) |
| TAID | Shing et al., arXiv 2501.16937, ICLR 2025 (spotlight) | The **target** is interpolated over time from the student's distribution toward the teacher's (adaptive schedule); TAID-LLM-1.5B, TAID-VLM-2B |
| Pre-training distillation | Peng et al., arXiv 2410.16215 | L = (1−α)·L_lm + α·L_kd. Tested Linear Inc (0→1), **Linear Dec (1→0): 39.2 avg, +4.1%** (beat Linear Inc, 38.1), Period (alternating 0 / 0.9), WSD-α (warm-up to 1, hold, cosine-decay to 0). Best: WSD-α + WSD-LR, +8.0%. Temperature: τ ≤ 2 similar; adaptive temperature methods not clearly better than static |
| TextBrewer (toolkit) | Yang et al., arXiv 2002.12620, ACL 2020 demo; textbrewer.readthedocs.io | `kd_loss_weight_scheduler` and `hard_label_weight_scheduler` with `linear_decay` (1→0) / `linear_growth` (0→1); `temperature_scheduler` (`constant`, `flsw`, `cwsm`). **A toolkit with KD/CE weight annealing has existed since 2020** |
| TRL GKD / GOLD / DistillationTrainer | TRL docs (main) | `lmbda`, `beta`, `temperature` are **fixed scalars**. In GKD, `lmbda` picks on- vs off-policy at random per batch. GOLD has `uld_crossentropy_weight` (default 0.0) and `uld_distillation_weight` (1.0), fixed. No schedule documented |
| DistillKit | README + `configuration.py` | Per-loss fixed `weight` and `temperature`; no schedule |
| torchtune | (verified 2026-11-01) | Fixed `kd_ratio` |
| Megatron-Bridge / ModelOpt | docs.nvidia.com/nemo/megatron-bridge/latest | Fixed `logit_kl_temperature` (default 1.0), `kd_loss_scale` (default 1.0), `skip_lm_loss` (default true); no schedule documented |
| ms-swift GKD | docs v4.0 | `sft_alpha` (default 0), `beta`, `lmbda` fixed; no schedule documented |

Precise conclusion:
- Annealing the KD/CE mix over training is **not new**: BAM 2019 (linear, KD share 1→0), Peng 2024 (Linear Dec, WSD-α, Period), and toolkit support in TextBrewer 2020. The thesis's cosine KL-ratio schedule and its optional oscillation (`kl_ratio_cycle_*`, cf. Peng's "Period") are variants.
- Temperature schedules are **not new**: Annealing-KD 2021 (decreasing), CTKD 2023 (learned curriculum), TextBrewer schedulers.
- Among the **current LLM toolkits** reviewed (TRL, DistillKit, torchtune, Megatron-Bridge, ms-swift, KDFlow README), none documents a schedule for the KD/CE ratio or the temperature. Safe wording: "the reviewed current LLM distillation toolkits use fixed loss weights and temperature; schedules exist in the literature [BAM, Peng] and in the older TextBrewer toolkit."

---

## 6. Toolkit update check, 2025–2026 (dates from GitHub releases / commits / docs)

**TRL** (GitHub releases API [V]):
- v0.25.0, 2025-11-06: experimental **GOLDTrainer** (cross-tokenizer, ULD-based).
- v0.26.0, 2025-12-09: GKDTrainer moved to `trl.experimental` (confirms the 11-01 note).
- v1.1.0, 2026-04-12: experimental `DistillationTrainer` (on-policy KD).
- v1.6.0, 2026-06-11: GOLD cross-tokenizer alignment via byte offsets.
- v1.8.0, 2026-07-09: GOLD VLM support; "**Raise a clear error when GKD student and teacher vocab sizes differ**" (#6252).
- v1.9.0, 2026-07-21: DistillationTrainer "**Validate the teacher by vocab size** instead of tokenizer identity" (#6457); `lmbda` removed from DistillationTrainer; IW-OPD objective.
- v1.10.0, 2026-08-13: `DistillationTrainer` graduates to **stable** `trl.DistillationTrainer`, plus a `trl distillation` CLI.
- v1.11.0, 2026-08-26: `DistillationTrainer` **supports tool calling**; new `AsyncDistillationTrainer` with multi-teacher (MOPD).
- v1.15.0, 2026-10-08: fused LM-head / Triton divergence kernel.
- v1.17.0, 2026-10-29: **X-Token** cross-tokenizer KD in GOLD; byte-fallback (Gemma) tokenizers in GOLD alignment.
- Docs [V]: DistillationTrainer teacher "must share the student's vocabulary". Loss = generalized JSD via `beta` (default 1.0 = reverse KL); no CE term. "Agent Training" section: the student calls Python tools during generation, and "Tool-result tokens are masked out of the loss".

**DistillKit** (GitHub [V]): last push 2026-05-12. 2025-12-15 "Update vLLM for logit capture" (vLLM ≥ 0.12). README unchanged on features: no vocabulary restriction (`d`/`vocab_size` = teacher vocabulary), no schedule, cross-tokenizer only via mergekit-tokensurgeon. README warning: "the configuration that was used to capture the logits must be reflected in the distillation configuration", which is a **manual** consistency rule. Code: the student loads via `from_pretrained`, embeddings are resized to the tokenizer size, and student logits are truncated to `true_vocab_size`. No content check of offline data against the vocabulary found.

**torchtune** (README [V]): "Torchtune is no longer actively maintained"; development "wound down in 2025". Last release v0.6.1, 2025-04-07; repo now `meta-pytorch/torchtune`. Successor not named on the README [UNVERIFIED: torchforge].

**EasyDistill** (GitHub [V]): AgentKD added (branch merged 2026-03-10: "virtual tool-use data synthesis and distillation"). **EasyDistill 2.0 released 2026-08-17**: a "config-driven rewrite" for **black-box** data synthesis (SFT / DPO / multimodal data, OpenAI-compatible endpoints), not backward compatible; v1.0 kept on a branch. The README lists `agent_distill` (LangGraph ReAct trajectories → SFT/DPO data) and `search_agent_distill`. No white-box logits, vocabulary or from-scratch features mentioned in the 2.0 README.

**NVIDIA**: Megatron-Bridge / ModelOpt KD [V] (GPT and Mamba providers; example Llama 3.2 3B → 1B; KL + optional cosine on intermediate layers + optional LM loss). NeMo RL X-Token off-policy guide [V] (Qwen3-4B → Llama-3.2-1B; DTensor V2 only; no version or date on the page, © 2026).

**New toolkit: KDFlow** [V]: Zhang et al., arXiv 2603.01875 (Mar 2026, v4 Sep 2026); GitHub README: off-/on-policy, cross-tokenizer, multi-teacher, KL/RKL/JS/AKL, SGLang teacher + FSDP2 student. Teacher logits are recomputed on the student side from the teacher's `lm_head` (full vocabulary). README news: "Accepted to the EMNLP 2026 System Demonstrations Track" (2026/08; self-reported, [UNVERIFIED] on ACL Anthology). No CE mixing / schedule, from-scratch, reduced vocabulary or data checks documented.

**ms-swift** [V] (v4.0 docs): GKD with Top-K renormalised mode, external teacher server, `sft_alpha`; reproduces Thinking Machines OPD with `--lmbda 1 --beta 1`.

**Axolotl** [UNVERIFIED]: KD from offline top-k logprobs reportedly added in v0.7.0 (early 2025, per a search snippet of the Axolotl Substack); docs URL returned 404.

**(a)–(d) matrix** (as documented, 2026-11-03):

| Tool | (a) reduced / task vocabulary | (b) tiny from-scratch students | (c) data ↔ vocabulary consistency check | (d) agent / harness training |
|---|---|---|---|---|
| TRL | No; loss-support restriction only (X-Token `xtoken_vocab_topk`, 2026-10-29) | Not documented; trainers accept a `PreTrainedModel`/`nn.Module`, but all examples are pretrained ≥0.5B; DistillationTrainer needs the teacher's vocabulary | Vocab-**size** check between live student and teacher (GKD 2026-07-09; Distillation 2026-07-21); nothing for stored data | **Yes, tool calling** in DistillationTrainer (2026-08-26), tool outputs masked; no checking/re-ask harness |
| DistillKit | No | No (`from_pretrained`) | Manual rule only (README); logits truncated to the true vocab size | No |
| torchtune | No | No | Not documented | No (unmaintained) |
| Megatron-Bridge / NeMo RL | No; P-KL top-k subset of the loss only | No (pretrained Llama examples) | Not documented | Not documented |
| ms-swift | No; per-position teacher Top-K renormalisation | Not documented | Top-K ≤ server `max_logprobs` [UNVERIFIED, v4.3] | Not checked |
| EasyDistill 2.0 | No | No (data generation for LLaMA-Factory / ms-swift) | Not documented | **Yes, agent trajectory synthesis** (AgentKD, 2026-03) as SFT/DPO data |
| KDFlow | No | Not documented | Not documented | Not documented |

---

## 7. Consequences for 3.1.6

### Must NOT be claimed as new
1. Logit distillation with stored (offline) teacher outputs, several divergences (FKL, RKL, JS, TVD), YAML/config-driven runs: TRL, DistillKit, torchtune, ms-swift, KDFlow.
2. Forward KL + CE as a weighted mix: Hinton 2015, Kim & Rush 2016, ULD (CE + λ·ULD), Multi-Level OT, DistillKit, torchtune, Megatron-Bridge.
3. **Annealing the KD/CE ratio** (BAM 2019 linear 1→0 KD share; Peng 2024 Linear Dec / WSD-α / Period; TextBrewer 2020 `linear_decay` / `linear_growth`), and **temperature schedules** (Annealing-KD 2021, CTKD 2023, TextBrewer). The kl-to-ce arm (1→0) is exactly Peng's "Linear Dec" and BAM's teacher annealing.
4. "KL beats CE": expected (section 4). Present it as confirmation in a new setting.
5. Renormalising the teacher over a token subset: Gemma 3 (256 sampled logits), ms-swift Top-K, DistillKit sparse KL with `ZERO` handling.
6. Restricting an output head to a frequency- or domain-derived subset of an existing vocabulary: Ushio 2023, VocabTrim 2025, FR-Spec 2025, TinyStories (top-10K tokens).
7. Distilling into students with a different or smaller vocabulary: cross-tokenizer KD (ULD, DSKD, MinED, MultiLevelOT, CDM, ALM, X-Token, GOLD) and reduced-vocabulary BERT students (Zhao 2019/2021, Kolesnikova 2022).
8. Students trained **from scratch** with logit KD in general: Gemma 2 (2B), Peng (1.9B), Busbridge (143M+), Anshumann (100M+), Kim & Rush (49–84M NMT).
9. "Tools assume the teacher's tokenizer": **outdated**. TRL GOLD (since 2025-11-06), X-Token in TRL (2026-10-29) and NeMo RL, KDFlow, and DistillKit via tokensurgeon all support other tokenizers.
10. "No tool supports agents": **outdated**. TRL DistillationTrainer tool calling (2026-08-26); EasyDistill AgentKD (2026-03).
11. Teacher-forced vs teacher-generated data as such (supervised KD vs SeqKD in GKD / TRL `seq_kd`).

### Plausibly new (wording: "not documented in the reviewed tools", checked 2026-11-03)
- A **task-derived output vocabulary** (a subset of the teacher's token ids, built from the task's teacher outputs) used as the **student's whole output head**, with teacher logits stored **dense over that subset** in bf16. The reviewed tools restrict at most the loss support (per-position top-K or global top-k), never the student's head.
- A **content fingerprint of that vocabulary stored with the teacher data and checked before training**. Reviewed tools check only vocabulary **size** between live models (TRL) or ask the user to keep configs consistent (DistillKit).
- Toolkit support for **tiny (≈1M–53M) task-specific students trained from scratch** inside the same framework as data generation and analysis. Research has from-scratch KD students (≥49M NMT; ≥100M LLM pretraining), and tiny reduced-vocabulary BERT students without logit KD.
- A **domain harness** around the distilled student (tag-routed actions, outputs checked against an executable solution / true places, one re-ask; harness-side fixed reply for out-of-scope), with own-history vs gold-history accuracy. TRL's tool-calling distillation masks tool outputs but documents no checking/re-ask harness or this evaluation pair.
- Built-in analysis pairing task accuracy with the majority baseline and seed cohorts with Welch + Bonferroni: not seen in the reviewed docs. I did not audit every tool's logging, so keep it soft.
- Annealing **inside a current LLM toolkit**: TRL, DistillKit, torchtune, Megatron-Bridge, ms-swift and KDFlow document only fixed weights and temperature. Claim it only as "the reviewed current LLM toolkits use fixed weights", never as a new idea.

### Corrections to the current talking points (thesis.md lines 342–346)
- L345 "They assume the student uses the teacher's tokenizer": change to "most require it; cross-tokenizer support exists (GOLD, X-Token, KDFlow), but for a different full tokenizer of a pretrained student".
- L345 "pretrained … usually 0.5B or more": true for toolkit examples (GKD Qwen2-0.5B; Distillation Qwen2.5-0.5B; GOLD 1B–1.5B; torchtune 1.5B→0.5B; Megatron-Bridge 3B→1B; TM Qwen3-8B-Base; KDFlow 30B-A3B→4B). Cross-tokenizer research goes down to 120–160M, also pretrained.
- L346 "Missing from all of them: … a check that the stored teacher data matches the vocabulary, an agent harness": narrow it to "a stored vocabulary fingerprint" (TRL has size checks) and "a checking / re-asking domain harness" (TRL has tool-calling distillation; EasyDistill has agent data synthesis).

---

## 8. UNVERIFIED list
- Kim & Rush 2016 students trained from scratch (not stated on the page; standard for 2016 NMT).
- Thinking Machines OPD: shared-tokenizer requirement (not stated; inferred).
- X-Token, ALM, DSKD: students "pretrained" (implied by the model names; not stated on the pages read).
- CDM student sizes (Qwen2-0.5B only from a search snippet).
- Busbridge et al.: students trained from scratch (implied by "distillation pretraining"); results of the top-p/top-k truncation section G.4 not read.
- Anshumann et al.: the exact sample count N (Appendix C not read).
- BiLD ">99% of mass in top-1024 logits" (search snippet only).
- arXiv 2608.03796: how the top-K tail is handled.
- KDFlow EMNLP 2026 demo acceptance (README self-report).
- torchtune successor name (torchforge); the README names none.
- ms-swift v4.3/v4.4 statements (`max_logprobs` default 20); only the v4.0 page opened (v4.4 URL returned 404).
- Axolotl KD (v0.7.0, offline top-k logprobs): only a search snippet; docs 404.
- EasyDistill EMNLP 2025 demo venue (search result only; arXiv 2505.20888 verified 11-01).
- Hinton et al. 2015 venue (NIPS 2014 Deep Learning Workshop): not checked; cite as arXiv.

## 9. Sources (all opened 2026-11-03 unless marked)
- ULD: https://arxiv.org/abs/2402.12030 (+ HTML v3)
- DSKD: https://arxiv.org/abs/2406.17328 (+ HTML v3); DSKD v2: https://arxiv.org/abs/2504.11426
- FuseLLM / MinED: https://arxiv.org/abs/2401.10491 (+ HTML v2)
- Multi-Level OT: https://arxiv.org/abs/2412.14528 (+ HTML v2)
- CDM: https://arxiv.org/abs/2502.11104 ; https://aclanthology.org/2025.findings-acl.419/
- ALM: https://arxiv.org/abs/2503.20083 (+ HTML v4)
- X-Token: https://arxiv.org/abs/2605.21699 ; NeMo RL guide: https://docs.nvidia.com/nemo/rl/latest/guides/xtoken-off-policy-distillation.html
- GOLD blog: https://huggingfaceh4-on-policy-distillation.hf.space/ ; TRL GOLD docs: https://huggingface.co/docs/trl/main/en/gold_trainer
- Thinking Machines OPD: https://thinkingmachines.ai/blog/on-policy-distillation/
- TRL GKD docs: https://huggingface.co/docs/trl/main/en/gkd_trainer ; DistillationTrainer docs: https://huggingface.co/docs/trl/main/en/distillation_trainer ; releases: https://api.github.com/repos/huggingface/trl/releases
- VT: https://arxiv.org/abs/2305.15020 (+ HTML v3); VocabTrim: https://arxiv.org/abs/2506.22694 ; FR-Spec: https://arxiv.org/abs/2502.14856
- Mixed-vocab BERT: https://arxiv.org/abs/1909.11687 (+ ar5iv); Russian KD: https://arxiv.org/abs/2205.02340 (+ ar5iv); TinyStories: https://ar5iv.labs.arxiv.org/html/2305.07759
- Sparse Logit Sampling: https://arxiv.org/abs/2503.16870 (+ HTML v2); Gemma 3: https://arxiv.org/html/2503.19786 ; TAD: https://arxiv.org/abs/2602.20816 ; BiLD: https://arxiv.org/abs/2406.13555 ; Offline top-K: https://arxiv.org/abs/2608.03796
- Hinton: https://ar5iv.labs.arxiv.org/html/1503.02531 ; Kim & Rush: https://arxiv.org/abs/1606.07947 (+ ar5iv); Gemma 2: https://arxiv.org/html/2408.00118 ; Peng: https://arxiv.org/abs/2410.16215 (+ HTML v1); Busbridge: https://arxiv.org/abs/2502.08606 (+ HTML v2); Wu: https://arxiv.org/abs/2404.02657 ; DistiLLM: https://arxiv.org/abs/2402.03898 ; Stanton: https://arxiv.org/abs/2106.05945 ; Yuan: https://arxiv.org/abs/1909.11723 ; GKD HTML: https://arxiv.org/html/2306.13649v3
- BAM: https://arxiv.org/abs/1907.04829 (+ ar5iv); Annealing-KD: https://aclanthology.org/2021.eacl-main.212/ (+ ar5iv 2104.07163); CTKD: https://arxiv.org/abs/2211.16231 ; TAID: https://arxiv.org/abs/2501.16937
- TextBrewer: https://arxiv.org/abs/2002.12620 ; https://textbrewer.readthedocs.io/en/latest/Configurations.html ; https://textbrewer.readthedocs.io/en/latest/Presets.html
- DistillKit: https://github.com/arcee-ai/DistillKit (README + source files `distillkit/lossfuncs/kl.py`, `missing_probability.py`, `configuration.py`, `main.py`, `trainer.py`)
- torchtune: https://github.com/meta-pytorch/torchtune ; EasyDistill: https://github.com/modelscope/easydistill (+ commits API)
- Megatron-Bridge KD: https://docs.nvidia.com/nemo/megatron-bridge/latest/modelopt/distillation.html
- ms-swift GKD: https://swift.readthedocs.io/en/v4.0/Instruction/GKD.html
- KDFlow: https://arxiv.org/abs/2603.01875 ; https://github.com/songmzhang/KDFlow
