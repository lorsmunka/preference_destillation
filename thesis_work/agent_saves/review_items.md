# review_items — adversarial review of math word-problem items + checking

Task: adversarial review of library/domain/math_tutor/{functions,items,renumber,prune,similarity,reword,harvest,benchmarks,validator_cases}.py and the item data (items.jsonl, renumbered.jsonl, benchmarks/); read-only, CPU-only.

## 2026-10-08 00:28 — save 1

Status: code read in full; most data checks run. Report not yet delivered.

### Findings so far

1. HIGH — VERIFIED — prune.py:215 (held-out split uses problem_shape only)
   Problem: an item goes to heldout_shape only when its TURN-0 shape is add>mul / sub>div / sub>add. Follow-up turns
   are not checked, so train items whose follow-up is one of those single-turn compositions stay in train.
   Spec (domain_specs.txt:115-116) says "only the single-turn composition is unseen".
   Evidence: train follow-up turns with a held-out shape: originals+rewordings add>mul 233, sub>add 68, sub>div 27
   (= 328); renumbered copies 633 + 176 + 52 (= 861). Total 1,189 training turns.
   e.g. S01-mem-b01-0082 follow-up `after_mall = state['now'] - 20 + 5` (sub>add);
   S03-mem-b04-0075 `shared = state['stickers'] - 15; each = shared // 6` (sub>div);
   S02-mem-b01-0044 `(state['sold_first'] + state['sold_second']) * 2` (add>mul).
   Fix: in prune's split, send a family to heldout_shape if ANY turn shape (item["shape"].split("|")) is held out
   (or drop those follow-ups), then rebuild items + renumber.

2. MEDIUM — VERIFIED — renumber.py:49-53 (replace_in_text, hyphenated compounds)
   Problem: text_numbers("ten-dollar") == [10], so the whole token "ten-dollar" is replaced by the new number word and
   the non-number half is lost.
   Evidence: 138 copies, e.g. S08-mem-b08-0031~n1 "He pays with 11 ten-dollar notes" -> "He pays with 20 thirteen notes"
   (code 11*10 -> 20*13; still passes validate, so nothing catches it). Also twenty-dollar 36, five-dollar 36,
   ten-kilogram 9, four-leaf 6, three-leaf 3, six-sided 3.
   Fix: in swap(), only replace the number half(s) of a hyphenated token and keep the rest ("thirteen-dollar"); or
   skip items whose slot value appears inside such a compound.

3. MEDIUM — VERIFIED — renumber.py:66-84 (slots_of / draw: no notion of fixed-value quantities)
   Problem: every written number is redrawn (half..double, <=99), including numbers whose value is fixed by the world.
   Evidence (renumbered.jsonl): 2,682 copies with a non-existent banknote ("$20 note" -> "$35 note"; originals only 14)
   — train 2,540 / test 116 / heldout 26; 136 copies with an impossible clock hour ("8 o'clock" -> "19 o'clock",
   "13 o'clock"; originals 0); >=145 copies turning a true fact false ("board has 4 wheels" -> "11 wheels",
   spider/insect legs, square sides...); 27 copies with a wrong ordinal suffix ("51th year", "7nd floor").
   Arithmetic stays consistent; the wording becomes false/absurd, concentrated in S09 "change from a note".
   Fix: treat numbers next to note/bill/o'clock/am/pm/ordinal suffix/legs|wheels|sides as fixed (like implied
   constants: skip the item or keep the value), or redraw denominations from {5,10,20,50} and hours from 1..12.

4. MEDIUM — VERIFIED — prune.py:213-226 + items.py:145 (family = item id, not the human seed)
   Problem: seeded items were written twice per seed (p1/p2 writers + r01), each its own family, so siblings
   reworked from the same ASDiv/MAWPS seed are split independently.
   Evidence: 2,152 seeds have >1 family; 402 seeds have a family in test AND one in train; 438 test items have a
   same-seed train sibling (all 438 with the same problem shape; text differs, median ratio 0.23). 28 share the exact
   turn-0 numbers: test "7 binders of 24 charts" (168) vs train "7 classes made 24 bookmarks" (168);
   test "58 girls and 67 boys" vs train "$58 and $67".
   Fix: family = source seed id for "seeded:*" items (so all reworks of one seed land in one split).

### Checked and fine (so far)
- functions.py interpreter: no exec; refuses calls, attributes, imports, loops, walrus, unary minus, **, bool/float
  literals, tuple/chained/subscript assignment, docstrings, nested defs, async def, defaults, mid-function return,
  bare `state`; 0..999 checked on every literal and every op result; exact // outside S07. Only crash: a 20,000-term
  `0+0+...` chain -> RecursionError (not NotAllowed) escapes validate (LOW, theoretical).
- items.validate re-run on 2,000 random items + 4,000 random copies: 0 failures, 0 answer/shape mismatches.
- No follow-up in items.jsonl inherits a stale answer (all set 'answer'; none put **state after 'answer'); every
  follow-up answer depends on state (taint check); no state key read after a same-named local (as_code unambiguous).
- Family/split: 0 families with mixed splits across items+copies; every copy has its original's split/family;
  0 held-out problem shapes in train/test; reword targets (10,847) all still train; no rewordings in test.
- similarity.WordSequenceIndex prefix filter: 0 mismatches vs brute force on 1,200 problems at 0.8 and 0.5.
- SVAMP leak re-check on final items (problem, and problem + first follow-up): 0 over threshold; 5 at 0.7-0.8.
- Name+number-swapped copies of all 1,000 SVAMP problems: all caught by the sequence test.
- Harvested seeds: grades all <=4 or null.

### Candidates / lower findings to write up
- LOW similarity.py:53-61: possessive names ("Mia's") are not normalised to NAME (token "mia's"); affects 60/1000
  SVAMP problems; did not let any simulated name-swap leak through (sequence test still catches).
- LOW (caveat) held-out pairs inside longer train shapes: train originals whose problem shape contains the pair
  contiguously: sub>div 800 (e.g. S07 add>sub>div round-up), add>mul 126, sub>add 54.
- LOW seeds vs SVAMP: 173/746 SVAMP test problems have a seed at sequence ratio >=0.6 (65 at 0.7-0.8, e.g. seed
  "Paco had 35 cookies. He ate 6" vs SVAMP "Paco had 40 cookies. He ate 2. Then he bought 37 more");
  SVAMP is built from ASDiv-A, so the seeds include its source problems; 107 items were reworked from those 53 seeds.
- LOW items.py validator gap: accepts a follow-up with no 'answer' key (inherits previous answer) or
  {'answer': x, **state}; data is clean today.

### Remains
- performance sanity of prune/renumber at full size (timing only, no rebuild)
- validator_cases coverage vs spec rules; reword.too_close vs prune behaviour; benchmarks check() edge cases
- implied-number logic in renumber (CONSTANT_WORDS) for false negatives

Scratch scripts: C:\Users\Ors\AppData\Local\Temp\claude\D--preference-destillation\ffe199e2-eaba-41f3-879f-8ba3f4ec1e4f\scratchpad\review_items\
  check_splits.py, check_renumber.py, check_renumber2.py, check_dataflow.py, check_reword_split.py, check_leak.py,
  check_seed_svamp.py, check_siblings.py, show_siblings.py, interp_tests.py, check_index.py, check_svamp_final.py
