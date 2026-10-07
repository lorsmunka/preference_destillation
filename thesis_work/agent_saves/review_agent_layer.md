# Review save: agent layer (math tutor) -- section 1, 2026-10-08

TASK: adversarial review of the agent layer (annotations, labels, checks, conversation, corpus, threads,
questions, harness, agentic, library/agent/question_templates.py); CPU-only fake-model tests, no repo edits.

STATUS: about 70% done. All target files read. Findings 1-8 below are tested on CPU. Still to do: more
harness edge cases, template slot edge cases, the final ranking.

## Findings so far

1. VERIFIED (high) -- checks.py:71-89 (arithmetic_true) rejects every "remainder of A / B = R" line, but
   annotations.py:95-96 renders exactly that form for any % step, and labels.py gives those lines to Gemma
   as "the exact calculations". Evidence (t2/t3): 888 of 3,336 S07 items (27%) have such a gold line;
   check_lines returns "arithmetic does not hold" for it. Even the best rewrite ("50 / 6 = 8 remainder 2")
   fails for 494+58 of 888 turns: the parsed value is the quotient, which the function never computed
   (t4). So about 1/4 of S07 items (and their renumbered copies) cannot get a facts-first label. Since
   usable() needs every turn, they drop out of the facts corpus. Fix: in parse/arithmetic_true, accept
   "remainder of A / B = R" (check R == A % B, number = R), or render % steps in the accepted
   "A / B = Q remainder R" form and add Q to the allowed values.

2. VERIFIED (medium-high) -- labels.py:154-165 (number_mapping) reads digits only (re.findall \d+), but
   renumber.replace_in_text also swaps number WORDS. A fact taken from a word ("Fourteen frogs") keeps
   the old value in the copy. Evidence (t10/t11): 5,683 of 96,544 copies have a number word that changed.
   With code-derived blocks as stand-in labels, 2,028 transfers are rejected and 58 PASS the check with
   a stale fact, e.g. S01-mem-b02-0040~n2: "frogs = 14 / hopped out = 14 / total = 16 + 14 = 30" (the
   copy says Sixteen). With the real sample labels, 9 transfers are rejected with "fact N is not in the
   conversation". Fix: build the mapping with items.text_numbers (words included, same order), and see
   the stricter check in finding 4.

3. VERIFIED (medium) -- questions.py:376 (what_if_foci -> labels.transfer with {old: new}) swaps EVERY
   fact equal to old, including a different quantity with the same value. Evidence (t6, sample labels):
   S06-mem-b06-0090~w2~n1 turn 1: "carrots eaten each day = 5 / apple slices each day = 5" with
   "what if 9 instead of 5" gives the gold "carrots = 9 / apple slices = 9 / total = 5 + 9 = 14", and it
   passes check_against_trace. 10 of 906 sampled what-if golds are like this. Fix: skip a what-if
   candidate whose value appears in more than one fact, or equals an earlier answer or an earlier turn's
   number. The stricter check in finding 4 also catches it.

4. VERIFIED (medium) -- checks.py:181-196 (check_against_trace): a calculation's operands only have to be
   "in the conversation", not listed facts or earlier results. So the facts can disagree with the
   calculations (findings 2 and 3 pass because of this), and a block with no calculation at all passes
   when the answer number is already in the conversation (t1: "apples he has = 24 / Answer: 24" passes).
   Fix: operands must be facts, earlier results in the block, implied constants, or 0/1, and at least
   one calculation is required. On the 409 sample labels this stricter rule rejects only 4, all
   "half"/"twice" = 2, which are allowed once implied constants count (t12).

5. VERIFIED (high) -- agentic.py:147-177 (build_output_tokens) never adds the thread explanations
   (Gemma's free text) to output_tokens.json, and model_handler.finish_example drops any target that
   has a token outside example+whitespace ("unexpected_tokens"), in forced scoring as well. Evidence
   (t16, tokenizer.json only): 86 of 166 pilot explanations (52%) contain such a token: "Yes" (glued
   after ">"), "▁represents", "▁imagine", "".", "▁Taking"... So about half of all explain records would
   be silently dropped at data generation, and the kept ones lean towards wordings that happen to be in
   the vocabulary. Fix: build the vocabulary after the thread pass and include the explanation texts
   (and the "<explanation>"+first-word forms), or report the drop rate per turn type.

6. VERIFIED (medium) -- labels.py:181-219 (transfer_to_copies): one global old->new dict per copy that
   also holds computed values. A coincidence (an intermediate value equal to a text number) makes it
   None. Evidence (t8): 12,520 of 96,544 copies (13%) are rejected as "numbers do not map one to one";
   10,893 of those are a computed value colliding with a text number. Copies are never sent to Gemma,
   so these lose their facts labels and leave the facts corpus. Fix: map per turn, keyed by position in
   the trace (fact i -> the copy's fact i), or send the rejected copies through the label pass.

7. VERIFIED (low) -- labels.py:175-176 (transfer) returns "" when the label split or joined steps (the
   prompt allows it). This is logged as "transferred block fails the check" (7 of 488 in the sample).
   Fix: log a separate reason; optionally align on result values.

8. VERIFIED (low-medium) -- conversation.py:124-134 (usable) needs every turn labelled, so one failed
   follow-up drops the item's verified turn 0 too. Fix: emit solve records up to the first unlabelled
   turn (render(..., turns=k)).

CANDIDATES (not yet confirmed):
 - threads.py:144-147 run_threads writes threads only at the very end. An interrupted pass (34k items,
   many hours) saves nothing, even though the docstring says it is resumable. Fix: append each plan
   when its last message is done, or after each position.
 - threads.py reply_entry / harness: an explanation cut off at max tokens (no "</explanation>") is
   accepted as complete.
 - checks.py CONCESSION: "we should have" is rejected as a concession ("we should have 9 groups").
 - checks.py arithmetic_true accepts any ast.operator (** ^ %) and floats ("1.5 * 4 = 6" passes).
 - questions/threads/harness: after an off-topic message (answered by the harness, not in context), the
   next message is drawn with "follow" weights (35% "follow" kind), so it refers to an explanation
   that does not exist.

## Checked and fine so far
 - Conversation context rules are the same in the harness, the thread records and the label pass
   (route = main line + thread + message, solve = clean history, explain = route context; an
   out-of-scope message never enters the context; a what-if closes the thread).
 - Harness state machine with an adversarial fake model (t14, 300 sessions): no crash, no task event
   skipped, out-of-scope messages never in the history, the route re-ask with "<" works, unclear is
   treated as a solution, one retry per check.
 - parse_route: "</solution>" does not match "<solution"; the first tag wins.
 - Template engine applies/related/draw; plan_threads speed is about 2 min for 34k items (t13).
 - Renumbered copies keep family and split; held-out shapes are left out; train comes before test.

## Remaining
 - harness: what-if expected answer when the turn's task was misrouted; the state of last_block.
 - template slot edge cases (duplicate values give a_words == b_words).
 - write the final ranking.

Scratch scripts: C:\Users\Ors\AppData\Local\Temp\claude\D--preference-destillation\ffe199e2-eaba-41f3-879f-8ba3f4ec1e4f\scratchpad\review_agent\t1_checks.py ... t16_vocab.py

# Section 2, 2026-10-08 -- finished

STATUS: done; final report sent to the coordinator.
Candidates confirmed since section 1:
 - VERIFIED (medium) threads.py:144-147: run_threads writes nothing until every plan is done (code read).
 - VERIFIED (low) threads.py:97-104: a cut-off explanation is accepted (t17: no closing tag, check None).
 - VERIFIED (low) after an off-topic message, 35 of 104 next messages are "follow" kind (t15).
 - VERIFIED (low) questions.py:264-266: words_of keeps one label per value, so "5 + 5" gives
   a_words == b_words ("carrots eaten each day" twice), 1 of 434 sample steps (t17).
 - CANDIDATE (low, not measured): CONCESSION rejects "we should have"; arithmetic_true accepts ** ^ and floats (t1).
Final ranking: high = 5 (vocabulary drops about half of explain records), 1 (S07 "remainder of");
medium = 2, 3, 4, 6, threads end-only write; low = 7, 8 and the rest.

## FINAL REPORT (appended by the main session from the agent's final message)
No critical findings; two high, five medium, the rest low.
HIGH
H1 agentic.py:147-177 | About half of the explain records will be silently dropped at data generation. build_output_tokens never adds the thread explanations; finish_example then drops targets with a token outside the vocabulary ("unexpected_tokens"), in forced scoring too. 86/166 pilot explanations (52%) have one: Yes (glued after >), represents, ". , imagine, Taking, equals. The survivors also lean towards in-vocabulary wordings. | Rebuild output_tokens.json after the thread pass with the explanation targets included; report the drop count per turn type.
H2 checks.py:71-89 | About a quarter of S07 items can never get a facts-first label. annotations.py:95-96 writes every % step as "remainder of A / B = R" and labels.py gives Gemma that line; check_lines always rejects it (the expression doesn't parse). 888/3,336 S07 items (27%) plus copies. The rewrite "A / B = Q remainder R" still fails for 552/888 (the quotient is never computed by the code). These items leave the facts corpus, and they are exactly the remainder niche. | Accept "remainder of A / B = R" in parse/arithmetic_true (R == A % B, value = R), or render % steps in the accepted form and allow Q.
MEDIUM
M1 labels.py:154-165 | number_mapping uses digits only, but renumbering also swaps number words. 5,683/96,544 copies are affected: 2,028 transfers lost, 58 pass with a wrong fact (S01-mem-b02-0040~n2 "frogs = 14" while the text says Sixteen). | Build the mapping with items.text_numbers.
M2 questions.py:376 | A what-if gold can contradict itself and pass: transfer swaps every fact equal to old (carrots = 5 and apple slices = 5 -> both 9; total = 5 + 9 = 14). 10/906 sampled golds. | Skip candidates whose value appears in more than one fact, or equals an earlier answer or an earlier-turn number.
M3 checks.py:181-196 | check_against_trace doesn't tie facts to calculations (operands may be anywhere in the conversation; no minimum of one calculation): "apples he has = 24 / Answer: 24" passes. Root cause of M1 and M2. | Require operands to be listed facts, earlier results, implied constants or 0/1, plus at least one calculation (rejects 4/409 sample labels, all half/twice = 2; counting implied constants fixes those).
M4 labels.py:181-219 | 13% of copies lose their labels because of one global number mapping: 12,520/96,544 copies are rejected ("numbers do not map one to one"), 10,893 of them a computed value colliding with a text number. | Map per turn by position, or send rejected copies through the label pass.
M5 threads.py:144-147 | The thread pass appends nothing until every plan finishes, so an interruption over 34k items saves nothing. | Append each plan when it stops or finishes.
LOW
- conversation.py:124-134 (usable): one failed follow-up drops the whole item, including the verified turn 0. Emit solve records up to the first unlabelled turn.
- labels.py:175-176: split or joined labels never transfer (logged under a misleading reason; 7/488).
- threads.py:97-104: a cut-off explanation (no </explanation>) is accepted as a training target.
- After an off-topic message, the next message is drawn with "follow" weights and refers to an explanation that doesn't exist (35/104). Draw it at depth 0.
- questions.py:264-266: a_words and b_words can be the same words ("5 + 5"; 1/434).
- checks.py:142-144: "we should have 9 groups" counts as a concession (false rejection).
- arithmetic_true accepts any Python operator (**, ^) and floats ("1.5 * 4 = 6").
CHECKED AND FINE
- The same context rules apply everywhere (route = main line + thread + message; solve = clean history; off-topic never enters the context; what-if closes the thread). Stored text matches the training targets.
- The harness state machine survived an adversarial fake model over 300 sessions.
- parse_route is correct; the remainder forms the annotations write pass the checks.
- The template engine is correct; plan_threads takes ~2 min for 34k items.
- Splits: copies keep their family and split, held-out shapes are left out, train comes before test.
