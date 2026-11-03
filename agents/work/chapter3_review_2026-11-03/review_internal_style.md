# Adversarial check of the rewritten chapter 3: own-work claims, writing rules, consistency

Checked 2026-11-03 against the repo `D:\preference_destillation` (read only). The draft is
`scratchpad/chapter3_draft.md`, the full thesis is `scratchpad/thesis_draft.md`, and the new text is the
">" lines of `scratchpad/chapter3_diff.txt`. I separated the truly new sentences from the user's sentences
that only got new citation numbers with a sentence-level diff (`scratchpad/new_sentences.py`, 84 new
sentences plus table rows). Only the new sentences were checked for style. Literature claims were skipped.

Primary sources I opened myself:
- `experiments/results/math_full.csv`, `experiments/results/math_agent.csv`
- `runs/*/info.json` (model_info)
- `git show cd807da:runs/tsc-scale-384h-12L-127M/...` (the April reddit scaling runs, not on disk)
- the safetensors headers in `~/.cache/huggingface/hub`
- `batches/*/info.json`, `library/domain/*/output_tokens*.json`
- the code under `library/`
- `thesis_work/running.txt` (STATUS), `thesis_work/findings.txt`, `thesis_work/idea_ledger.txt`

I also re-ran a CPU-only inference check (16.5M student, GPU hidden, `python -B`, nothing written to the repo).

Severity labels:
- **WRONG**: the claim contradicts the evidence.
- **IMPRECISE**: true only with a caveat the text leaves out, or worded more strongly than the evidence allows.
- **STYLE**: a writing-rule or grammar issue.
- **CONSISTENCY**: the claim clashes with another part of the thesis.

---

## Part A. Own-work claims

### A-1. "harmincszor–kétezerszer kevesebb paraméter" (3.1.2), "harmincszor–kétezerszer kisebbek" (3.2). Numbers PASS, IMPRECISE as worded

Teacher sizes, counted from the safetensors headers:
- Gemma 4 12B: 11,959,730,224 parameters; the language-model part alone is 11,907,350,320.
- Gemma 3 4B: 4,300,079,472 parameters (index total_size 8,600,158,944 bytes / 2).

Students that beat the most-common-answer baseline, by domain:

| Domain (teacher) | Student | Params | Task acc. vs baseline | Ratio | Source |
|---|---|---:|---|---:|---|
| Math tutor (Gemma 4 12B) | math-full-kd-s1 | 6,106,798 | .340 vs .041 | **1,958x** | math_full.csv row 2 |
| Math tutor | math-full-kd-m-s1 | 16,389,422 | .495 vs .041 | 730x | math_full.csv row 5 |
| Math tutor | math-agent-kl/ce-m-s1..3 | 16,456,494 | .677–.719 vs .042 | 727x | math_agent.csv pass 10 |
| Math tutor | math-full-kd-l-s1 | 53,732,910 | .580 vs .041 | 223x | math_full.csv row 6 |
| Reddit (Gemma 3 4B) | tsc-scale-384h-12L-127M (April) | 129,186,573 | .874 vs .636 (parseable replies only) | **33x** | git cd807da, eval_epoch 2 |
| Reddit | tsc-reduced-input-proj-5M (April) | 4,932,493 | .863 vs ~.64 (parseable only) | 872x | internal_evidence 1.2; thesis 2.5.5 |

Students that did not learn their task:
- math tutor 2.7M (.050 vs .041)
- the E-01 reddit and math v1 students (at the class prior, findings.txt:296-307)
- the coding-agent pilots (task accuracy 0)

Post-generation has no task accuracy.

- The 2,000x end is real: the 6.1M tutor student against the ~12B Gemma 4 12B (1,958x; 1,950x against the language-model part alone).
- The 30x end is real but weak:
  1. It is an April 2026 reddit run that the thesis never presents (only "méretskálázási futásai" in 4.5).
  2. The run predates the 2026-10-08 evaluation fixes and has one seed.
  3. Its accuracy counts only the 8,179 replies that parsed out of ~10,000 test comments. I recomputed the baseline from its confusion matrices: 63.6%. If unparseable replies count as wrong, accuracy is ~71% against 63.6%. It still beats the baseline, but by ~8 points, not ~24.
  4. 100,663,296 of its 129.2M parameters (77.9%) are a *full* 262,144-token input embedding. The 30x end therefore exists only because this student did *not* use the reduced input vocabulary that 2.5.1 and 3.1.5 describe.

**Fix, option A** (keep the range and make it traceable): "...a feladatukat megtanuló tanulómodellek a tanármodellnél nagyjából harmincszor–kétezerszer kevesebb paraméterből álltak (a szöveghangulat-elemzésnél kb. 33–870-szer, a matematikai ügynöknél kb. 220–2000-szer)." Then present the April scaling runs in chapter 9 with the parseable-only caveat.

**Fix, option B** (agents only, the stronger evidence): "...a matematikai ügynök feladatát megtanuló tanulómodellek a tanármodellnél nagyjából 220–2000-szer kevesebb paraméterből álltak."

### A-2. "tízszer–nyolcvanszor kisebbek (6–54 millió paraméter)" (3.1.4), "6 és 54 millió" (3.1.5), table "kb. 6–54 millió". PASS
- 0.5B / 53,732,910 = 9.3x and 0.5B / 6,106,798 = 81.9x, so "nagyjából tízszer–nyolcvanszor" holds.
- The working tutor students are 6.1M, 16.4M, 16.5M and 53.7M (math_full.csv, math_agent.csv).
- The 2.7M student is correctly left out.

### A-3. Output vocabularies 525 / 19 502 / 11 007. Numbers PASS; the coding-agent clause is IMPRECISE
- **525 (sentiment): PASS.**
  - runs/e01-reddit-*/info.json `output_vocab_size` 525
  - the April run tsc-scale-384h-12L-127M `output_vocab_size` 525 (git cd807da)
- **19 502 (math tutor): PASS.**
  - The full-run domain reads `library/domain/math_tutor/output_tokens_full_v1.json` (full.py:78-80; 19,426 example tokens). Adding the whitespace, prompt and auxiliary sections (vocabulary.py:142-203) gives 19,502.
  - The same number appears in batches/math_tutor_full_v1/google_gemma-4-12B-it/info.json `vocabulary_size` 19502 (sha1 7bcb36cd) and in every math-full-* and math-agent-* info.json `output_vocab_size` 19502.
  - The "18,357" in CLAUDE.md and the 18,962 in output_tokens.json belong to the slice. They are not used by the reported students.
- **11 007 (coding agent): number PASS, claim IMPRECISE.**
  - The number: `library/domain/code_agent/output_tokens.json` (10,883 tokens) plus the shared sections gives 11,007 (batches/code_agent/google_gemma-4-12B-it/info.json `vocabulary_size` 11007, sha1 63558d75).
  - The claim: **no student was ever trained on it.** The only coding students are pilots on the 3,498- and 7,771-token pilot vocabularies, and they learned nothing (running.txt "END TO END 2026-10-21"; internal_evidence 2.5).
  - So "és ez alkotja a tanulómodell teljes kimeneti rétegét" is true by design but untested for the coding agent.

**Fix (3.1.5)**: "A kimeneti szótár a tanármodell saját tokenjeinek a domainhez kiválasztott részhalmaza (lásd 2.5.1. alfejezet), és ez a részhalmaz alkotja a tanulómodell teljes kimeneti rétegét. A szöveghangulat-elemzésnél 525, a matematikai ügynöknél 19 502 tokenből áll, a kódkereső és -magyarázó ügynök tanítóadatai pedig egy 11 007 tokenes szótárral készültek."

### A-4. 16,5M student, 61,7%; full vocabulary 140,8M and 95,5%. PASS

Configuration of runs/math-agent-kl-m-s1/info.json: hidden 256, 6 layers, 4 heads.
- Input embedding: 20,116 × 256 = 5,149,696 (the 20,116 rows include the unknown row).
- Output head: 19,502 × 256 + 19,502 bias = 5,012,014. The head has a bias and is **untied** from the input embedding (transformer.py:106, 118).
- Body: 6,294,784 (attention 1,572,864, FFN 4,718,592, norms 3,328; no bias in attention or FFN, transformer.py:310-314, 382-384).
- Total: 16,456,494.

Recomputed shares:
- Embedding + head = 10,161,710 / 16,456,494 = **61.749%**, i.e. 61,7%.
- With Gemma 4's 262,144 tokens, untied, same body: 67,108,864 + 67,371,008 + 6,294,784 = **140,774,656** (140,8M), with embedding + head = **95.53%**. Gemma 4's `embed_tokens` is [262144, 3840]; the transformer prints "reduced from 262,144".
- If tied, as Gemma itself is, the model would be 73.7M.

Optional precision: "ugyanez a modell (külön be- és kimeneti mátrixszal)".

Consistency note: 2.5.4 (user text) calls the 195,980-reply students "Ugyanaz a 16,4 millió paraméteres tanulómodell", but they are 16,456,494 = 16,5M. The 16,4M (16,389,422) is the first-loop kd-m student, which has a different input vocabulary (19,854 vs 20,116) and different loss settings. See C-1.

### A-5. 3.3 numbers. All PASS; the wording is IMPRECISE in four places

- 2,7M student 5,0% vs baseline 4,1%: **PASS** (math_full.csv row 3: 0.0497 / 0.0414; 18 vs 15 of 362 answers). One seed, 64,554 train replies.
- 6,1M student 34%: **PASS** (row 2: final 0.3398, best 0.3508; kd 0.9→0.1). One seed, 64,554 replies.
- KL better than CE at all 30 paired measurements: **PASS**. I counted it from math_agent.csv: 30/30, with the smallest margin +0.35 points at seed 3, pass 1. The arms are pure KL (kl_ratio 1.0) and pure CE (0.0). Pairing holds: the same seed gives the same start weights and data order (commit 3e89b4e; experiments/math_ce_vs_kl.py:5).
- Final means 70,5% vs 68,3%: **PASS**. These are the pass-10 values averaged over 3 seeds:
  - KL .700 / .719 / .697: mean .7052, sd .012
  - CE .684 / .688 / .677: mean .6829, sd .006
- "a tanítás végén átlagosan": correct (last pass, mean over seeds). It is ambiguous only because the CSV holds seeds × passes. Write "a 10. epoch után, a három ismétlés átlagában".

Problems with the wording:
1. **Overclaim, "a dolgozat ezt ... erősíti meg".**
   - The 30 measurements are 3 independent pairs. The 10 passes of one run are not independent.
   - The framework's own test (significance.welch_t_test, re-run by me) gives **p = 0,067** on the final pass (t 2.86, df 2.9).
   - A paired t-test, computed by hand and not available in the framework, gives p = 0,036 (t 5.11, 2 df).
   - With 3 seeds and the framework's own test above 0.05, "erősíti meg" is too strong.
2. **The setup is missing.** The paragraph moves from 2.7M/6.1M students (64,554 replies) to the KL/CE result without saying that the latter is the 16,5M student on 195,980 replies. A reader will assume it is the 6.1M students. An n=1 kd-vs-ce pair at 6.1M (.340 vs .304) also exists, which adds to the confusion.
3. **Redundant**: "három ismétlés mind a 30 páros mérésénél (10 epoch, 3 ismétlés)" says 3 ismétlés twice. "három ismétlés mind a ..." also lacks an article.
4. **"oldotta meg helyesen"**: task accuracy compares the student's `Answer:` with Gemma's answer (TutorAnswerMetric, agentic.py:75-81; teacher_mode forced). This matches the thesis's own definition in 2.1.7 ("a tanármodell válaszához mérve"). Gemma is right ~94-97% of the time (running.txt), so the gap is small, but strictly it is agreement with the teacher.

**Fix (3.3, replaces the last three sentences of the paragraph)**: "A matematikai ügynöknél 64 554 tanítóválaszon, egy ismétlésben a 2,7 millió paraméteres tanulómodell még nem tanulta meg a feladatot (5,0%-os feladatpontosság a mindig a leggyakoribb választ adó megoldás 4,1%-ával szemben), a 6,1 millió paraméteres viszont a tesztfeladatok 34%-ánál már a tanármodellel egyező választ adott. A 16,5 millió paraméteres tanulómodellnél 195 980 tanítóválaszon a tiszta KL-veszteség mindhárom ismétlésben és mind a 10 epoch végén jobb feladatpontosságot adott a tiszta CE-veszteségnél, a 10. epoch után a három ismétlés átlagában 70,5%-ot a 68,3%-kal szemben (lásd 9. fejezet). A különbség kicsi, három ismétlésnél a páros t-próba szerint p ≈ 0,04, a keretrendszer Welch-próbája szerint p ≈ 0,07. Ez a szakirodalom alapján várható irány, és a dolgozat egy, a feldolgozott szakirodalomban nem vizsgált méretben és beállításban ugyanezt az irányt mutatja."

### A-6. Framework features (3.1.6, 3.3). All PASS, with two small caveats

| Feature | Evidence (file:line) | Verdict |
|---|---|---|
| The whole output head is the task subset | library/model/student_vocabularies.py:36-37 (`output_vocab_size = vocabulary['vocab_size']`); library/model/transformer.py:118 (`nn.Linear(hidden_dim, output_vocab_size)`); the subset is a list of Gemma token ids, vocabulary.py:142-203 | PASS |
| Teacher logits stored at every position over all subset tokens | data_gen/model_handler.py:331-347 (forced mode: `logits[:, :, reduced_ids]` for every target position); :384 (generated mode, every step: `extract_logits_as_vector`); shared/vocabulary.py:213-218 | PASS |
| Vocabulary fingerprint stored and checked | shared/batch_files.py:42-45 (sha1 of the ordered token list), 151-152 (written into every batch); data_gen/saving_handler.py:15, 30; data_gen/generator.py:48, 77 (info.json), 88-91 (refuses to append with a changed vocabulary); training/batch_handler.py:121-122, 161-168 (trainer refuses a mismatched batch); model/student.py:94-99 (checked when a run is loaded) | PASS. Caveat: batches written before the fingerprint existed pass unchecked (batch_handler.py:164), which covers the April reddit data |
| Harness that checks and re-asks | domain/math_tutor/harness.py:1-30 (route by first tag; solution and explanation each checked and asked again once; out_of_scope answered by the harness); math_tutor/demo.py:21-34 (`ask`, retries=1); domain/code_agent/harness.py:1-12 (scope and explanation checked, asked again once) | PASS |
| Task accuracy next to the most-common-answer baseline | training/evaluator.py:34-45 (`majority_baseline`), 159-163 (logs `majority_baseline`, `task_accuracy_excess`) | PASS |
| Teacher-forced vs own-history accuracy | training/evaluator.py:140-141 (both token-level, every eval); tooling/analysis.py:38-39, 179-180 | PASS at token level. Conversation-level own history (the student's own earlier turns) exists only in experiments/slice_trial.py and is not measured for any working student. The text does not claim conversation level, so it is fine |
| Statistical tests between seeds (Welch + Bonferroni) | tooling/significance.py:25-43; tooling/run.py:398-409 (`Experiment.compare` uses Welch); Bonferroni applied in experiments/anneal_direction.py:217 | PASS (no paired test in the module, see A-5) |

### A-7. "videokártya nélkül is futtathatók" (3.2). PASS for the tutor students, IMPRECISE in scope and traceability
- The code supports a CPU fallback:
  - library/shared/device.py:7-12 and library/model/student.py:28-33 (cuda, then mps, then cpu)
  - the checkpoint is loaded with `map_location=device` (student.py:107)
- My re-run: scratchpad/work/cpu_timing.py, `CUDA_VISIBLE_DEVICES=-1`, one thread, i7-13700K.
  - math-agent-kl-m-s2 (16,456,494 parameters) ran through the real harness turn (`demo.take_turn`: route, solution, check).
  - 4 word problems, all answered right, in **0,33–0,51 s per turn**, 243 tokens/s; process memory 976 MB (Python + PyTorch + tokenizer).
  - The earlier evidence sheet reports 1,0–1,5 s per turn for the 53,7M student.
- Caveats:
  - The measurement is not in the repo, so a reviewer cannot trace it.
  - Nothing was measured for the 129M reddit student (plausible, not shown).
  - The coding agent has no student at all.

**Fix (3.2)**: "...így videokártya nélkül, egy asztali processzoron is futtathatók (a matematikai ügynök 16,5 millió paraméteres tanulómodellje egy processzormagon kb. 0,3–0,5 másodperc alatt válaszolt egy feladatra, lásd 8. fejezet)." Add the measurement (script + numbers) to chapter 8 or 9.

### A-8. Other own-work statements in the new text

1. **IMPRECISE, borderline WRONG: "Az eszközök az utóbbi időben a dolgozat két fő irányát is elérték."** The next sentence names these two directions as cross-tokenizer distillation (ULD) and tool-calling students. Cross-tokenizer distillation is *not* a direction of the thesis:
   - the student uses the teacher's own tokenizer (student_vocabularies.py:18, `AutoTokenizer.from_pretrained(teacher_model)`);
   - its output tokens are a subset of Gemma's ids (vocabulary.py:185-188).

   The thesis's direction is a smaller vocabulary drawn from the same tokenizer.

   **Fix**: "Az eszközök az utóbbi időben két, a dolgozathoz közeli iránnyal is bővültek." Optionally add: "A dolgozat tanulómodellje viszont a tanár tokenizerét használja, csak a szótára szűkebb, ezért eltérő tokenizerek kezelésére nincs szükség."
2. **IMPRECISE: "néhány millió paraméteres, alapoktól tanított generatív tanulómodellbe" (3.3).** The working agent students are 6–54M, and the headline one is 16,5M. "néhány millió" undersells and contradicts 3.1.4, 3.1.5 and the table (6–54M). **Fix**: "...kerül desztillálásra egy 6–54 millió paraméteres, alapoktól tanított generatív tanulómodellbe...".
3. **IMPRECISE: "ezért fontos mérni, hogy a tanár valószínűségének mekkora része marad a szótáron belül (lásd 2.5.1. alfejezet)" (3.1.6).**
   - 2.5.1's "kb. 99%" rests on one ledger note (thesis_work/idea_ledger.txt:34-38): Gemma 3 4B, part-1 domains, no script, no CSV.
   - The in-vocabulary mass is not logged anywhere in library/ (no logsumexp over the full row).
   - It was never measured for Gemma 4 12B or the agents, whose 19 502 / 11 007 vocabularies are the ones discussed here. The only proxy: 413 of 204,808 tutor replies were dropped for an out-of-vocabulary token (0.2%).

   The text says "fontos mérni" and then points to a measurement that does not cover the agents. Either measure it before submission (your memory reminder "retained mass") or add: "A dolgozatban ez eddig csak a szöveghangulat-elemzésnél, a Gemma 3 4B-vel készült el, a két ügynöknél még nem."
4. **IMPRECISE (anachronism): "Elsősorban ez a hiány indokolta a saját keretrendszer elkészítését".**
   - The framework existed in March 2026 (run tsc-scale-384h started 2026-03-23).
   - The documentation check is dated 2026-11-03.
   - TRL's cross-tokenizer (Nov 2025) and tool-calling (Aug 2026) support came after parts of the framework.

   **Fix** (justification in the present tense): "Ez a hiány indokolja a saját keretrendszert (lásd 4. fejezet), amely..."
5. **IMPRECISE (minor): "amely modellhámmal együtt ügynökként működik" (3.3).** True: the student runs inside the harness (demo.py, tutor_chat.py; my CPU run used `take_turn`). But no harness-level evaluation of a student exists (route accuracy, re-ask rate, whole conversations). Task accuracy is per reply, with Gemma's history. Fine for chapter 3; chapter 9 should not claim more.
6. **IMPRECISE (minor): "a konfigurációs fájlokkal vezérelt tanítás" (3.1.6).** The framework is configured with typed Python objects in a run list (`TrainingRun(...)` in RUNS, train.py:1-25; library/specs.py), not with separate config files. **Fix**: "a konfigurációval vezérelt tanítás".
7. Table row "A dolgozat", column "Cél": "egy domain, modellhámmal ügynökként". The sentiment domain is not an agent. Optional fix: "egy domain, az ügynököknél modellhámmal".

---

## Part B. Writing rules (new text only)

### B-1. Colons: PASS
The 84 new sentences contain no colon. The only colons in ">" lines are in the user's own sentences, which changed only their citation number:
- "Előnye, hogy nem szükséges újratanítani a modellt: ..."
- "Ezenfelül a domainre hangolás ... is mehet: ..."
- "Az ügynökök desztillálása már létező kutatási terület: ..."
- 4.1 "ezt ki is használja: ..." and "fedezni tudja: ..."

### B-2. First person singular: PASS
None. The new text is impersonal or passive. The first person plural that remains ("tanítjuk", "lemondunk") is in the user's own sentences.

### B-3. Terms and English parentheticals

First mentions, searched over the whole thesis_draft.md:

| Term | First mention | Verdict |
|---|---|---|
| SLM, small language model | line 329 (3.1.4) | first mention, OK |
| fingerprint | line 343 | first mention, OK |
| tool calling | line 340 | first mention, OK |
| GKD / generalized knowledge distillation | line 339 | first mention, OK, but see S-3 |
| ULD / universal logit distillation | line 340 | first mention, OK, but see S-3 |
| top-k | defined at 2.3.2 (line 242) | not repeated in the new text, OK |
| logit | defined at 2.1.3 | not repeated, OK |
| offline | already used bare at lines 6, 18 and 378, meaning "without internet" | see S-2 |

- **S-1 (STYLE): "ismétlés" is never defined.** Its first mention in the whole thesis is new text (3.1.6, line 343, "az ismétléseket pedig statisztikai próbával"). 2.3.4 talks about "több futtatás". Your term list pairs ismétlés with seed, so define it here. **Fix**: "...az azonos beállítású, de eltérő véletlen kezdőértékkel (seed) indított ismétléseket pedig statisztikai próbával veti össze."
- **S-2 (STYLE): "előre elmentett (offline) tanárlogitokkal".** "offline" already appears bare three times in the thesis with another meaning. The parenthetical both repeats an existing word and switches its sense. **Fix**: "Az Arcee DistillKit [48] előre elmentett tanárlogitokkal dolgozik".
- **S-3 (STYLE): "a GKD (generalized knowledge distillation) módszert", "az ULD (universal logit distillation) veszteségre".** Unlike the thesis's pattern ("magyar kifejezés (english term)"), no Hungarian term is given. **Fix**: "megvalósítja az általánosított tudásdesztilláció (generalized knowledge distillation, GKD) módszerét [47], amelyben a tanulómodell..." and "az univerzális logitdesztillációs (universal logit distillation, ULD) veszteségre [51] építve".
- **S-4 (STYLE): two names for one term.** The new text uses "kis nyelvi modell" (lines 329, 333, 355, 362) and "kis méretű nyelvi modell" (the 3.1.5 title and its first sentence). 1.2 has "kis méretű nagy nyelvi modellek" (user text). Pick one. Following the section title: "kis méretű nyelvi modellek (small language model, SLM)", "Kis méretű nyelvi modellen a szerzők...", "A legtöbb kis méretű nyelvi modell...", and in 3.2, 3.3 and the table "a kis méretű nyelvi modellek".
- **S-5 (STYLE): "előre tanított szóbeágyazásokat" (3.1.1).** Elsewhere the thesis says "előtanított". **Fix**: "előtanított szóbeágyazásokat használó BiLSTM-hálózatot".
- **S-6 (STYLE, low): "Ügynökdesztilláció"** (table, 3.2, 3.3) vs the 3.1.4 title "Ügynökdesztillálás". Optional: use "ügynökdesztillálás" in the new text.
- Terms checked and correct: tanármodell / tanulómodell, modellhám (no repeated "(harness)" in new text), feladatpontosság, "helyes előzménnyel mért" / "saját előzménnyel mért" pontosság, epoch, "mindig a leggyakoribb választ adó megoldás". One drift: 3.3 says "a leggyakoribb választ adó megoldáshoz mért feladatpontossággal" without "mindig", while the next sentence has it. **Fix**: "a mindig a leggyakoribb választ adó megoldáshoz mért feladatpontossággal".

### B-4. Grammar, spelling, numbers
- **S-7: "Kim és Rush [27] gépi fordításon azt is megmutatta"** → "megmutatták". With two people as subject the plural reads better and matches "Hinton és társai ... fogalmazták meg". The possessive "Kim és Rush ... fordítómodelljében" is correct as it stands.
- **S-8: "legalább néhány száz millió paraméteres" (3.3)** → "legalább néhány százmillió paraméteres" (the numeral is written as one word).
- **S-9: "a dolgozat ezt egy a feldolgozott szakirodalomban nem vizsgált méretben"** → "egy, a feldolgozott szakirodalomban nem vizsgált méretben". A comma is needed before the expanded attributive phrase, as in the sentence before it ("egy, a feldolgozott szakirodalomban nem szereplő beállításban").
- **S-10: the 3.1.5 vocabulary sentence has a dangling instrumental** ("...részhalmaza (...), a szöveghangulat-elemzésnél 525, ..., 11 007 tokennel, és ez alkotja..."). See the A-3 fix, which also repairs the grammar.
- **S-11: "Nem szerepel a tanítóadatok mellé elmentett szótár-ujjlenyomat (fingerprint) sem, amely ellenőrzi, hogy..."** The fingerprint does not check anything itself. **Fix**: "..., amellyel ellenőrizhető, hogy a tanítóadatok és a tanulómodell ugyanazzal a szótárral készültek-e."
- **S-12: "A tanulómodellek tehát a TinyStories méretében"** (TinyStories is a dataset/paper, and 54M is 5x its "<10M"). **Fix**: "A tanulómodellek tehát a TinyStories-modellekhez hasonló méretűek, és alapoktól tanulnak, de egyetlen feladatot sajátítanak el, a tanármodell szótárra szűkített eloszlásából."
- S-13 (optional): "kerülnek elmentésre" (twice in 3.1.6) is a wordy construction. Possible: "ha csak a tanár legvalószínűbb tokenjeit mentik el", "és a tanár logitjait minden pozícióban ennek a részhalmaznak az összes tokenjére elmentik". The user's own text uses the "kerül + -ásra" form too, so this is optional.
- S-14 (optional): the new text uses "viszont" 8 times and "tehát" 5 times. Some could go.
- Numbers all correct:
  - decimal comma (0,96; 90,7%; 61,7%; 140,8; 95,5%; 5,0%; 4,1%; 70,5%; 68,3%)
  - space as thousands separator for 5+ digits (19 502, 11 007, 262 144), with four-digit "5135" unspaced as the thesis does elsewhere (1345, 8210)
  - suffixes: "-os/-ával/-át/-a/-ot/-kal/-szer/-öt" all agree
  - "harmincszor–kétezerszer" and "tízszer–nyolcvanszor" spelled correctly with an en dash
  - "11 billió" = 10^12, correct and consistent with "2,8 billió" in 1.1
  - "2026. november 3-i állapot" correct

  The separators are plain spaces; a non-breaking space in Word would keep "19 502" on one line (optional).
- Caption "3.1. táblázat. A tömörítési és desztillációs módszerek összevetése": PASS. It matches the "2.1. ábra. ..." style (number, period, title, no final period) and sits above the table, which is the Hungarian convention for tables.

### B-5. Cross-references
- "(lásd 2.5.1. alfejezet)" ×2 → 2.5.1 "A redukált szótár korlátai" builds the vocabulary and states the 99% retained mass: PASS.
- "(lásd 3.1.5. alfejezet)" in 3.1.3 → 3.1.5 shows pretrained models stop at ~135M: PASS.
- "a 3.1.4. alfejezetben említett Belcak" → yes: PASS.
- "(lásd 3.1.2. alfejezet)" in 3.2 → 3.1.2 says a distilled model can be quantized: PASS.
- "(lásd 4. fejezet)" → 4.1 lists the framework's steps: PASS. "(lásd 4.1. alfejezet)" would be more exact.
- **"(lásd 9. fejezet)" points at an empty chapter.** It must hold the KL vs CE per-pass curves and the paired numbers before submission (CONSISTENCY reminder).
- "A 3.1. alfejezetben bemutatott módszerek" and "A 3.1. táblázat": PASS.

### B-6. Citations: PASS (checked programmatically over the whole draft)
- 58 reference entries, numbered 1–58 without gaps.
- Every cited number has an entry, and every entry is cited.
- First mentions in the body run exactly 1, 2, ..., 58. [14] is first cited in 2.1.1, so its reuse in the 3.1 table is fine.

---

## Part C. Consistency with the other chapters and within chapter 3

- **C-1 (CONSISTENCY): 16,4M vs 16,5M.**
  - 2.5.4 says "Ugyanaz a 16,4 millió paraméteres tanulómodell ... 64 554 ... 49,5% ... 195 980 ... 68–72%".
  - 3.1.5 calls the 195,980-reply student "16,5 millió".
  - Both size labels are right for their own runs (16,389,422 and 16,456,494). But they are not "ugyanaz": they differ in input vocabulary (19,854 vs 20,116), loss (kd 0.9→0.1 vs pure KL/CE) and test set (362 vs 3,147 answers).
  - Suggested wording for 2.5.4 (yours to write): "Egy hasonló méretű (16,4, illetve 16,5 millió paraméteres) tanulómodell...".
- **C-2 (CONSISTENCY): the coding agent's status.** Chapter 3 now carefully says only "a dolgozat matematikai ügynökének működő tanulómodelljei". Other chapters imply a working distilled coding agent, but none exists:
  - the abstract (line 9: "kettőnél ... önálló ügynököket hoz létre")
  - 1.1 (line 18: "laptopon is lokálisan futtatható kódmagyarázó ügynök")
  - 4.2 (line 377: "képes ... megtalálni és ... megmagyarázni")

  Only Gemma plays the coding agent. 3.1.5's "11 007 ... alkotja a tanulómodell teljes kimeneti rétegét" adds to the impression (fix in A-3).
- **C-3 (CONSISTENCY): phone claim.** 1.1 (line 18) says "okostelefonon is offline elfut". Chapter 3 says only "videokártya nélkül" (true, desktop CPU). Nothing in the repo measures a phone.
- **C-4 (CONSISTENCY): "five domains" (abstract).** Of the five domains, only two have students that clearly learned the task: the math tutor, and the April reddit runs with the caveat in A-1. Math v1 and the E-01 reddit students sit at the class prior (X-C1), and post-gen has no task accuracy. Chapter 3's "a feladatukat megtanuló tanulómodellek" ratio covers sentiment and the tutor only. This is not a contradiction, but chapter 9 must say which domains the ratio is drawn from.
- **C-5 (CONSISTENCY): reduced input vocabulary.** 2.5.1 says "A dolgozatban vizsgált tanulómodellek be- és kimeneti szótára ... redukálva van". The student behind chapter 3's 30x end (and the math v1 and post-gen E-01 students) used the full 262,144-token *input* vocabulary (git cd807da info.json `input_vocab_size` 262144; runs/e01-math-*, e01-postgen-*).
- **C-6 (CONSISTENCY, within chapter 3): what "előtanított" covers.**
  - 3.2: "a TinyStories apró, előtanítás nélküli modelljei". But the table row "Kis nyelvi modellek [41], [42]" says "alapoktól, általános előtanítással", and 3.3 lumps "a kis nyelvi modellek" with "jellemzően előtanított, legalább néhány száz millió paraméteres" students.
  - 3.2 and 3.3: "az általános desztilláció ... előtanított ... tanulómodellekkel". But the table says "alapoktól vagy előtanított modellből", and 3.1.1 says Gemma 2 and 3 were trained by distillation (from scratch).
  - 3.1.1: Tang's BiLSTM uses "előre tanított szóbeágyazások". But the table row "Korai feladatspecifikus desztilláció" says "előtanítás nélkül", and 3.3 says "Kis, előtanítás nélküli modellekbe".

  Fixes:
  - 3.2: "A korai feladatspecifikus desztilláció és a TinyStories apró, alapoktól tanított modelljei..."
  - 3.2: "...az általános desztilláció, az ügynökdesztilláció és a desztillációs eszközök pedig jellemzően milliárdos nagyságrendű, általános célú vagy előtanított tanulómodellekkel dolgoznak."
  - 3.3: "...vagy jellemzően legalább néhány százmillió paraméteres, általános célú tanulómodellel és teljes szótárral dolgoznak, mint az általános desztilláció, az ügynökdesztilláció, a legtöbb kis méretű nyelvi modell és a desztillációs eszközök..."
  - Table cell (Tang / Kim & Rush): "alapoktól, legfeljebb előtanított szóbeágyazásokkal".
- **C-7 (CONSISTENCY, low): Gemma 3's 256 sampled logits now appear three times.**
  - 2.3.2 (user)
  - 3.1.1 (new: "A Gemma 3 család modelljeit már mind desztillációval tanították, ... 256 mintavételezett logitot ...")
  - 3.1.6 (new: "A Gemma 3 desztillációjánál ez tokenenként változó, mintavételezett részhalmaz volt [12]")

  Optional fix: in 3.1.6, "A Gemma 3 desztillációjánál ez tokenenként változó, mintavételezett részhalmaz volt (lásd 3.1.1. alfejezet)".
- **C-8 (CONSISTENCY, low): vocabulary size notation and term.** 4.1 (user) says "a 256 ezer tokenes szókincs" (Gemma 3 270M), while 3.1.5 says "262 144 tokenes szótár". These are the same 2^18 = 256 × 1024 size, written two ways, and "szókincs" vs "szótár" (the term defined in 2.1.2). Possible fix in 3.1.5 (new text): "A Gemma teljes, 262 144 (256 × 1024) tokenes szótárával...".
- **C-9 (CONSISTENCY, low): redundancy.** "6–54 millió" appears three times in a row (3.1.4, 3.1.5, the table). Optional: drop it from 3.1.5's first paragraph ("A dolgozat tanulómodelljei nagyrészt még az alsó határnál is kisebbek (lásd 3.1.4. alfejezet).").
- **No contradictions found** on:
  - Gemma 4 12B "közel 12 milliárd" (2.1.1) vs the 2,000x ratio
  - "billió" (1.1 and 3.1.5)
  - the 4.1 cost and training claims
  - 2.5.4's "68–72%" (it covers 67.7–71.9; rounding to "68–72" is fine)

---

## Claims verified as correct (one line each)
- 2,000x end: 11,959,730,224 / 6,106,798 = 1,958x (math_full.csv row 2; Gemma 4 12B safetensors header).
- 30x end: 4,300,079,472 / 129,186,573 = 33x, a student that beats the baseline (.874 vs .636 on parseable replies; ~71% vs .636 if unparseable count as wrong).
- 6–54M working tutor students; 9.3x–82x below 0.5B.
- Output vocabularies 525 (reddit info.json), 19 502 (math batches + runs, sha1 7bcb36cd), 11 007 (coding batches, sha1 63558d75).
- 16,5M = 16,456,494; embedding + head 61,749%; full vocabulary untied 140,774,656 with 95,53%.
- 2,7M .0497 vs baseline .0414; 6,1M .3398 final.
- KL > CE at 30/30 (seed, pass) pairs; pass-10 means .7052 vs .6829; same start weights and data order per seed.
- Head = subset (student_vocabularies.py:37, transformer.py:118).
- Logits stored for every subset token at every reply position (model_handler.py:331-347, 384).
- Fingerprint stored per batch and in info.json, checked by the generator, the trainer and the loader.
- Checking and re-asking harness for both agents (math_tutor/harness.py, code_agent/harness.py).
- Majority baseline and excess logged every eval (evaluator.py:34-45, 159-163).
- Teacher-forced and own-history token accuracy logged every eval (evaluator.py:140-141).
- Welch + Bonferroni between seed cohorts (significance.py, run.py:398-409).
- CPU fallback in code; 16,5M student 0,33–0,51 s per harness turn on one CPU core (re-measured today).
- No colons and no first person singular in the new sentences.
- Citations 1–58 in first-mention order, all cited, all present.
- Cross-references 2.5.1, 3.1.2, 3.1.4, 3.1.5 and "4. fejezet" point at the right places; "9. fejezet" is still empty.
- Caption style "3.1. táblázat." matches "2.1. ábra."; Hungarian number formats and "%-os" suffixes correct.
