[TALKING POINTS - Claude, 2026-11-04. Raw material for chapter 6, gathered by 5 read-only agents: git history (493 commits), the earlier TDK rounds (2025-10 log, 2026-01 thesis, 2026-03/04 slides), the working docs (running.txt, findings.txt, ledgers, code review), the results CSVs, and code comments. Copy: agents/work/chapter6_points_2026-11-04/talking_points.md. Pick, do not use all; replace with your own text, then delete this block.]
Main angle: 14 months, 8 phases. The method stayed the same; what changed was how it is measured and how reliable the pipeline is. Agents were the motivation from the first abstract (2026-01: "ágensrendszerek (döntési pontok)") but became the goal only in 2026-10.

6.1 Munkafázisok (table candidate: phase | time | what was built | what it taught)
- 1. Prototype, 2025-09 → 10 (67 commits). Reddit comments → JSON labels with Gemma 3 4B, first through Ollama, after 6 days through HF transformers (f05f9e6 → 0d42049; reason inferred: per-step logits). First own student 470M, of which 268M input embeddings (full 262k vocabulary), output vocabulary 93. Training accuracy ~.96, own generation .216. One run ~97 h; a Chinchilla-scale run was estimated at ~39 days on the 4070 (908f320).
- 2. Rebuild, 2026-01 → 02 (49). One-day rewrite on 01-20: packages, GeGLU / RMSNorm / RoPE, KL/CE anneal. First thesis text ("Domain Specifikus LLM Desztillációs Pipeline"). HUN-REN A100-20C server, ~2x slower than the local 4070.
- 3. More domains + first TDK round, 2026-03 → 04 (98). Math word problems, output vocabulary from data (reddit 525 tokens), reduced input vocabulary (03-24; 42.72M → 6.02M parameters at 80h/5L), sizes 5M-198M, post-generation domain. TDK talk plan (03-24), slides (04-28), thesis draft ch. 1-12 (04-21/22). Claimed then: output-vocabulary reduction is "the main innovation", "with no performance loss".
- 4. Library restructure, 2026-06 → 08 (14). library/ package, Analysis facade, typed run configs (specs.py, 07-15), one script per experiment instead of the queue JSON (08-26).
- 5. Hardening, E-01, Gemma 4 12B, 2026-09 (47). bf16 .npz batches, dashboard + Stop, Gemma 4 12B in NF4 on the 12 GB card after re-sharding. E-01: 100 runs, "no effect".
- 6. Math tutor + code review, 2026-10-04 → 08 (95). Corpus of 34,002 items, template pupil, AI code review (58 findings, 1 critical = X-C1), test suite born (23 → 115 tests, 157 by 10-31), vocabulary fingerprint.
- 7. Diagnosis + coding agent, 2026-10-12 → 21 (57). Slice trial 0/54 → diagnosed as data-limited. Coding agent (10-20). Both agents end to end locally.
- 8. Rented GPU + results, 2026-10-25 → 11-04 (66). RunPod: two math data loops and the coding-agent data (~$40 + ~$45.50 + ~$39, about $125 in all; check before quoting). Size sweep, 3-seed KL vs CE, custom student networks.

6.2 Tapasztalatok - measurement (the biggest lessons)
- A score at the baseline is a floor, not a null. Code review 2026-10-08 (X-C1): reddit students gave the most common label 93-100% of the time, accuracy .621-.629 against a .626 baseline; math numeric answers 5/284. So E-01's 100-run "no effect" proved nothing, and the April TDK table (reddit 64-65%, math 22-23%) was already at the floor. Since then the baseline and the excess are logged at every eval (findings.txt 296-333; evaluator.py:156).
- Token accuracy hides the class prior. A prior-only student still scores .885 token accuracy, +.003 over its baseline (ideaA_vocab_sweep.csv). The 2026-01 pilot's 87.69% was this kind of number.
- Teacher-forced vs own generation. 2025: ~.96 training accuracy vs .216 own generation. Now both are reported; for the agents, own-history next to gold-history.
- One fixed test set. The "data plateau" bump (66.7% at 1,024 examples) was its own test set's baseline (.671): every size had a different test set. Now the test set is a fixed tail (T-H1).
- One seed is a direction, not a result. KL vs CE needed 3 paired seeds: +2.2 points, KL ahead in all 30 paired tests; the seed spread (1.2 points) is as large as the gap.
- Loss and task accuracy can disagree. Size sweep: 16.4M had lower test loss than 53.7M (.782 vs .848) but answered worse (.495 vs .580).
- Invalid outputs used to drop out of the denominator, overstating accuracy; now they count as wrong (metrics/task.py:3-6).

6.2 Tapasztalatok - data
- Data-limited, not epoch-limited. Slice trial (2026-10-14): 1,345 replies, 5.9M student, 0/54 solved (most common answer 4/54), while 98-100% of replies were well-formed. Learning curve: test CE -0.34 per doubling of replies, not flattening, while each run flattens by pass 8.
- A toy fix may not carry over. One update per batch instead of per example: 99.9% vs 45% on a copy task, yet the real trial stayed at 0/54.
- New problems beat renumbered copies: test error 1.48 vs 1.59 (slice_mix.csv).
- More data crossed the floor (2026-10-26, 64,554 records): 6.1M .340, 53.7M .580 against .041. Capacity threshold: 2.7M stayed at .050 (one seed).
- Leakage found and fixed: split by item, not by family (402 seeds on both sides), held-out shapes checked on the first turn only (1,189 training turns) → family = human seed; rebuilt the same day, 0 leaks.
- Training data = Gemma's own replies, mistakes included, instead of gold targets (2026-10-13).
- (Probably chapter 9) Generalisation is bound to shapes: two-step problems .40 → .17 on unseen shapes; SVAMP .43.

6.2 Tapasztalatok - engineering
- Vocabulary fingerprint. Rebuilding the vocabulary reorders the logit columns silently (KL targets land on wrong tokens, CE still looks fine) → sha1 stored and checked. It paid off on 2026-10-16: four old students were refused.
- A stale resume point produced a corrupted run marked "completed" → the step must match, the resume file is used once (trainer.py:5-11).
- Crash safety: a partial last batch overwrote a full one (Virág lost 32 of 5,000 examples) → write first, count after, atomic renames.
- Teacher quirks. Gemma 4 sometimes opens an empty thought channel (dropping those replies cost 5.1%, so it is tolerated narrowly). Gemma 3's tokenizer adds BOS, Gemma 4's does not. Re-feeding the prefix gave worse logits under bitsandbytes than KV-cached decoding.
- NF4 and bf16 teachers give different numbers → the setup is stored per data file and mixing is refused.
- Windows limits. A 24 GB shard cannot be memory-mapped (error 1455) → 2 GB shards. The GPU silently spills into system RAM (8 replies per call 20-90 s, 4 per call ~2 s). Hidden windows slow training from 0.70 to 1.09 s per batch.
- Long inputs: fused attention cut a 6,000-token input from 24 GB to 3 GB; batched KV-cached rollouts were 13-20x faster (they had taken 64-74% of the time).
- Fun fact: the optimizer state goes only into the resume checkpoint, 65.9 MB vs 197.6 MB (AdamW keeps two values per parameter).
- solve() is never given to exec: an AST interpreter; the review agent's escape attempts were all refused.
- Known bugs survived across rounds: CE scaled by T² flagged 2026-03-13, fixed 10-08; per-example updates visible in 2025, fixed 10-14. An audit list is not a fix.

6.2 Tapasztalatok - cost and infrastructure
- Estimates collapsed: ~39 days (2025) → 2,870 h (2026-01) → $300-686 (2026-03) → the full tutor data for ~$40. Inefficient code, not hardware, drove the old numbers.
- Where it ran: 4070 → HUN-REN A100-20C (~2x slower) → 4070 → RunPod A100 80 GB and RTX PRO 6000.
- Five students side by side on one A100: ~4x throughput (one alone left it ~75% idle).
- Estimates ran optimistic: $31 → $40 (only Gemma's pass had been timed), 14 GB → 18 GB.
- The untested self-stop failed (runpodctl refused the pod's own key) → stop through the GraphQL API, tested on a throwaway pod; all three runs then stopped themselves.

6.2 Tapasztalatok - direction and way of working
- Claims that moved. "Output vocabulary reduction with no performance loss" (2026-03) was never tested against a full-vocabulary student; "525 tokens carry ~19x the information" was wrong (reddit teacher entropy exp 1.02). The 2026-10-08 novelty check found the task-derived output vocabulary is no longer new (VocabTailor). Now the claim is the combination (3.3).
- Loss schedule: KL first (2025) → KL .99 → .30 / .50 (spring 2026) → E-01 post-gen: pure KL best (.305 > ... > pure CE .264).
- "Nem framework" (2026-04 slides) → "Keretrendszer" in the title now.
- Ideas that only promised a better student: 28 checked, 28 dead (idea ledger, 2026-08-31). Deferred: second teacher, Muon, multi-token prediction, Hungarian tutor.
- Code finder dropped on 2026-10-05, back as the coding agent on 10-20, without edit actions ("so complicated and doesn't add more").
- Way of working: since 2026-09-15, 243 of 265 commits are co-written with Claude Code and the decisions are logged (running.txt, findings.txt); AI review agents found 58 issues in one day, fixed the same day with tests. Ties to 3.1.6.
