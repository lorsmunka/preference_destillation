# thesis_positioning save file (append-only)

## 2026-10-08 save 1 (literature pass done, page not yet written)
TASK: adversarial review of the thesis's value/novelty -> thesis_work/thesis_positioning.html (local page; red team,
why worth doing, novelty + low-hanging fruit, framing, sources). Read-only on repo otherwise. No git, no GPU.
STATUS: all reading done (CLAUDE.md, thesis_description, novelty_check, findings, running, domain_specs,
template_pilot, harness_explainer, code: losses.py, model_handler.py, generate_data.py, output_tokens.json).
~60 sources opened. thesis_positioning.html: NOT WRITTEN YET (confirmed absent 2026-10-08).

### Project facts gathered (for the page)
- Final tutor data planned in bf16 on cloud 5090 (generate_data.py load_in_4bit=False); NF4 only pilots + Viràg.
- batches/*/info.json records no teacher precision. CLAUDE.md still says teacher Gemma 3 4B / ~525 tokens (drift).
- Tutor output vocab 18,357 tokens (output_tokens.json); input vocab 11,286; teacher tokenizer 262,144 BPE.
- score_training_examples already logs teacher_agreement (argmax == gold share); full logits in memory there
  and in ReducedLogitsRecorder -> in-vocab mass is a ~10-line add.
- No pytest suite (validator_cases.py 24/24 exists). Assignment text requires tests + "telefonon futó" demo.

### Objections (severity -> best answer -> closer)
1 Why not fine-tune pretrained 135M/270M? HIGH -> assignment says from scratch; TinyGSM random-init ~-10 pts;
  Krishna 2023 self-pretraining; 3.4M vs 270M; BUT Gemma 3 270M INT4 runs on Pixel -> add SmolLM2-135M SFT row (~1 day).
2 KD adds nothing (own E-01 null). HIGH -> own it: null at exp(H)~1, strict order at 2.73; Menon theory;
  measure exp(H) per turn type + write prediction (1 h, no training).
3 LLM-written + template pupil. HIGH -> templateness 0.40 vs SVAMP 0.91, 0 leaks, held-out shapes, free-written
  175/180; weakness: free-written also LLM-written (Li 2022) -> human-written routing test, templated vs free gap.
4 4-bit teacher. LOW-MED -> final data bf16; NF4 vs bf16 on 256 turns on the cloud box; QAT blog is no defence of NF4.
5 SVAMP easy/contaminated. MED -> contamination is a pretrained-model issue (GSM1k), 0 leaks; real issue 72.5% UNK;
  check SVAMP gold facts-lines vs 18,357-token output head.
6 A classifier could route. MED-HIGH -> LIDSNet/CLINC150; agent = route+solve+explain+state; BoW LR router baseline (2 h).
7 Too small to explain. HIGH -> TinyStories fluency, faithfulness open; number-grounding check; size study on explain.
8 No human eval. MED-HIGH -> LLM-judge biases (Zheng); 40-conv blind rubric, kappa.
9 One agent domain. MED -> part 1 has 4 domains/7 entropy points; turn types = within-domain axis.
10 Toolkits duplicate. MED -> TRL GKD, DistillKit (offline compressed logits!), EasyDistill; feature table.
11 Reduced vocab not new + biased. MED -> TinyStories, VT, Compact, VocabTailor (task-derived static head, training-free);
  Sparse Logit Sampling bias -> log in-vocab mass.
12 Statistics. MED -> measured noise floor; pre-register primary comparisons; 5 seeds on headline row.
13 Error compounding. MED -> MT-Eval gold vs self history; Laban 2025; gold-vs-own-history eval (3-4 h).
14 Pedagogy/safety. MED -> Bastani PNAS 2025, MathDial; EU AI Act Annex III 3(b) if it evaluates learning; limitation.
15 Assignment compliance: tests + phone. HIGH(grading) -> pytest wrapping existing checks; CPU/phone latency table.

### Novelty verdicts
(a) from-scratch tiny domain agent w/ token distributions: NOVEL as combination (LLM4Teach, SAD, Kang, STAR, TinyGSM).
(b) task-derived reduced head + KL restricted: PARTLY (VocabTailor 2508.15229 contradicts novelty_check line).
(c) trace-grounded follow-ups: PARTLY (MathCAMPS counterfactual follow-ups w/ solver; MalruleLib; MathDial; Schatzmann).
(d) per-turn-type entropy regimes: PARTLY NOVEL (SAD span losses; EGAD token entropy) - within-agent test not found.
(e) gold vs own history eval: NOT NOVEL as method (MT-Eval), new measurement here; cheap.
(f) teacher-forced soft labels on verified gold: NOT NOVEL (Kim & Rush word-level KD; Knowledge Adjustment).
(g) harness answers OOS: NOT NOVEL (NeMo Guardrails, CLINC150).
(h) teacher prompt / student none = context distillation (Snell 2022): cite, not claim.
(i) first tag = route: Self-RAG reflection tokens: cite.
(j) anti-sycophancy at tiny scale: PARTLY (Wei 2023, FlipFlop) - free metric.
(k) teacher-vs-gold disagreement map: NOT NOVEL method (KA), free measurement (teacher_agreement exists).

### Draft framing
Claim: a modular framework distils a 12B teacher into from-scratch students of a few M params; soft labels
matter in proportion to teacher entropy (4 domains + within one agent). C1 framework, C2 entropy rule, C3 tutor case study.
Avoid: "task vocab is new"; "helps/safe for children"; "from-scratch/KD beats pretrained/SFT".

### Sources opened (links)
arxiv.org/abs/: 1503.02531 1606.07947(ar5iv) 2005.10419 2106.05945 2306.13649 2604.00626 1911.07471 2209.15189
2305.07759 2312.09241 2209.14389 2408.00118 2410.16215 2502.08606 2402.14905 2502.02737 2502.10645 2311.13373
2505.13820 2505.17612 2602.03022 2609.36630 2506.02153 2310.11511 2310.10501 2607.08268 2305.15020 2508.15229
2509.06836 1909.11687 2503.16870 2605.01732 2510.24021 2103.07191 2405.00332 2410.05229 2407.00900 2305.14536
2405.19444 2601.03217 2410.12294 2308.03958 2311.08596 2505.06120 2401.16745 2112.08812 1909.02027 2110.15717
2410.01627 2306.05685; aclanthology N07-2038, 2024.emnlp-main.478, D16-1139; PNAS via ideas.repec (Bastani);
MLSys 2021 Bouthillier; Gemma 3 270M + QAT blogs; TRL GKD docs; github arcee-ai/DistillKit; gdpr-info.eu art-8;
artificialintelligenceact.eu annex 3; mlanthology MathCAMPS (NeurIPS 2024 MATH-AI workshop); PMLR busbridge25a.

### Remaining
Write thesis_positioning.html (TL;DR, red-team table, why worth doing, novelty table, framing, sources); final report.
