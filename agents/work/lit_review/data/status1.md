# Literature review agent -- save file (append-only)

**Task:** literature review groundwork for the BSc thesis: build thesis_work/literature_review.html (TL;DR, annotated bibliography by theme, positioning table, chapter outline) and thesis_work/literature.bib, every reference verified.

**Status (all 11 themes researched and verified; writing outputs):**
- Done: A KD foundations; B KD for autoregressive LMs; C pre-training distillation / from-scratch; D small models on synthetic data + vocabulary; E distilling reasoning; F agents / harnesses / agent distillation; G structured output / routing; H maths word-problem datasets; I tutoring + sycophancy; J error compounding / multi-turn; K toolkits.
- Verification method: arXiv API metadata (title, authors, date, comments) for every arXiv entry; venues cross-checked with arXiv comments, Semantic Scholar, ACL Anthology, PMLR, icml.cc; key details read from full text (Peng 2025, Gemma 2/3, TinyGSM, MT-Eval, LLM4Teach, SAD, Kang, Octopus v2, EasyDistill, TextBrewer docs, TRL GKD docs, DistillKit README).
- Output files: literature_review.html NOT YET WRITTEN; literature.bib NOT YET WRITTEN (both checked absent at start). Content lives in the scratchpad (scripts/entries.py) and will be generated from one source.

**Unverified / caveats so far:**
- TinyGSM App. B random-init numbers (53.1 vs 63.3 at 125M; 55.5 vs 65.9 at 350M; 57.3 vs 68.2 at 1.3B) are pass@1 without verifier according to a full-text fetch summary; confirm in the PDF table.
- LLM4Teach and Structured Agent Distillation: student initialisation not stated in the papers.
- Gemma 2: which teacher distils the 2B model not confirmed (summary said 27B teaches 9B; not relied on).
- Harness-Zero student model (Qwen3.5-9B) only from secondary sources; not used.
- Claim seen in a blog snippet that "the harness explains ~8x more variance than the model": NOT verified, do not cite.
- torchdistill / DistillKit / KDFlow: schedule support not documented (marked "not stated").
- Structured Agent Distillation DOI from Semantic Scholar (10.65109/OLHJ8062) not opened; omitted from the bib.

**Draft TL;DR points:**
1. LLM KD moved from word/sequence KD (2016) to divergence choice and on-policy KD (2023-25); 2025-26 work explains hard+soft mixing (Bridge-Garden, ICML 2026) and stored-logit bias (Sparse Logit Sampling, ACL 2025). Students are pretrained 0.1-7B.
2. From-scratch logit distillation exists only at general pre-training scale (Gemma 2/3, Peng et al. ACL 2025); tiny from-scratch models on teacher text (TinyStories, TinyGSM) use CE on text only.
3. Agent distillation 2024-26 (Kang NeurIPS 2025, SAD AAMAS 2026, STAR ICLR 2026, TinyAgent, Octopus, roadmap 2609.36630, Harness-Zero) uses pretrained 0.5-9B students.
4. Thesis = combination: random-init 1-100M student + task-derived output vocabulary + stored teacher logits over it + annealed KD/CE and temperature + eval-time distribution analytics + model+harness agent on executed-code-verified data.
5. Do NOT claim: annealed KD->CE (BAM 2019 teacher annealing; LLM4Teach; Peng WSD schedule; TextBrewer weight schedulers), entropy-aware per-token weighting (Bridge-Garden, EGAD), reduced vocabulary per se (Kolesnikova, VocabTrim, TinyStories), stored truncated logits (Peng, Gemma 3, DistillKit, EasyDistill), first-token routing (Octopus functional tokens).
6. Biggest exposures: teacher-forced and gold-history metrics overstate (Arora; MT-Eval 7.32 vs 4.64); capacity gap (Zhang ACL 2025); no pretrained-student baseline (TinyGSM: random init costs ~10 points); entropy axis has 2-3 points and is predicted by theory (Menon; Bridge-Garden).

**What remains:** write the generator for HTML + BibTeX, positioning table, chapter outline, risks; validate both files; final save section.
