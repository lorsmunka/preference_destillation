# Writing follow-up message templates (math tutor)

## What this is for
A small model is being trained to be a primary-school maths tutor. After it solves a word problem
(facts first, then named calculation lines, then `Answer: N`), the pupil sends follow-up messages.
We need **thousands of templates** for those messages. A program fills each template's `{slots}`
from a real solution, so one template becomes many messages. The tutor model must learn, from the
message's **meaning**, whether to explain (`explanation`), to solve again (`solution`, only for
"what if" changes), or that the message is off-topic (`out_of_scope`, the harness answers with a
fixed sentence).

A solution the pupil sees looks like this (one conversation turn):

```
<task>After the farmer's 4 cousins brought over some hens, the coop held 27 instead of 19. How many hens did the cousins bring?</task>
<solution>
hens before = 19
hens after = 27
hens the cousins brought = 27 - 19 = 8
Answer: 8
</solution>
```

Lines like `hens before = 19` are **facts**; `hens the cousins brought = 27 - 19 = 8` is a **step**.
The 4 is a **distractor** (in the story, rightly not used).

## Format: one JSON object per line
```
{"kind": "why", "focus": "step", "requires": ["subtract"], "depth": "first", "text": "how come you took {b} away from {a}?"}
```
- `kind`: one of the kinds below (its route is fixed by the kind; you may omit `route`).
- `focus`: what the message is about; decides which `{slots}` you may use.
- `requires` (optional): features the focus must have. **This is how you write niche templates**:
  a template that requires `["divide", "rounded_up"]` is only used on rounded-up division steps,
  and niche templates are preferred whenever they apply.
- `depth`: `"first"` (opens the questions after a solution), `"follow"` (only after the tutor has
  explained something), `"any"`.
- `text`: the pupil's message. `{slot}` is replaced by a value. **No digits** in the text: every
  number must come from a slot.

## Rules
1. **Voice**: real children aged 7-10 typing to a tutor. Short. Vary a lot: polite, blunt,
   confused, cheeky, shy, chatty; some lowercase without punctuation; some with "um", "wait",
   "ok but"; a few very short ("why {b}??"). Do not start many templates with the same words.
   No adult or teacher vocabulary ("operation", "quantity", "subtrahend").
2. **Grammar must hold for every fill.** Use `{operation_phrase}` / `{operation_phrase_past}` /
   `{other_phrase}` for an operation with its numbers ("take 9 away from 54"); if you want
   operation-specific wording ("why did you share {a} into {b} groups"), add `"requires": ["divide"]`.
   Word slots (`{step_words}`, `{words}`, `{a_words}`...) are noun phrases taken from the solution
   ("hens the cousins brought", "price of a tetra"); write around them so any such phrase reads well
   (`"the {words}"`, `"the bit about {step_words}"`).
3. **Niche**: at least 40% of your templates should use `requires` (one or two features), spread over
   many different features. Story-type and money/time features let you speak the story's language
   ("is that the change you get back?", requires money).
4. **No near-duplicates**: each template must differ in more than one or two words from your others.
5. **Run the check and fix everything** it reports, then read its example fills and rewrite any
   that read badly:
   `python -m library.domain.math_tutor.questions check library/domain/math_tutor/questions/bank/<your file>.jsonl`
   (it prints each template filled on real conversations, and how often it applies; a niche
   template that applies to 0% may be fine if its condition is real but rare).
6. Write **only your own file** (create it; it must not exist yet). Do not edit any other file,
   do not run git, do not use the GPU, no web access needed.

## Kind-specific notes
- `pushback`: the pupil insists the tutor is WRONG and gives their own answer `{wrong_answer}`
  (computed as a plausible mistake; features say which: `wrong_operation`, `used_distractor`,
  `skipped_step`, `slip`). Vary the authority and confidence ("I'm sure it's...", "my mum says",
  "the answer sheet says", "no way it's {answer}", "my friend got...").
- `what_if`: the pupil asks for a **new answer** with one number changed: `{old}` -> `{new}`
  (`{old_words}` names it when the solution does). It must clearly ask for a new result.
  Questions about a different *method* ("what if you added instead?") are NOT what_if (they are
  `wrong_operation` / `another_way`).
- `follow`: reacts to the tutor's last explanation, whose words you cannot see: generic
  ("that makes no sense", "can you give me an example?", "ohh so it's like sharing?") or about a
  focus with `depth: "follow"`.
- `off_topic` (focus `none`, no slots): anything that is not about this maths: other school
  subjects, questions about the tutor itself, games, pets, jokes, requests for stories, wanting to
  stop, mild rudeness or silliness. A **new maths problem is not off-topic** (it is a task).
  Keep it child-appropriate.
- `distractor`: about the unused number `{distractor}` (or its sentence `{distractor_sentence}`).


## Kinds (kind -> route)

- `why` -> `explanation`: why the tutor did a step (this operation, these numbers, this order)
- `meaning` -> `explanation`: what a number or a step stands for in the story; often as a guess in the story's words, asking if it is right
- `where_from` -> `explanation`: where a number came from (the problem, an earlier answer, a word like 'pair')
- `check` -> `explanation`: whether a line or the answer is really right, or how the tutor knows
- `another_way` -> `explanation`: whether a step or the whole problem could be done another way
- `lost` -> `explanation`: the pupil does not understand a step and asks for it more simply
- `shortcut` -> `explanation`: whether there is a quicker way
- `smaller_numbers` -> `explanation`: to see a step again with smaller, easier numbers
- `wrong_operation` -> `explanation`: why the tutor did not use another operation here (why not take away, why not multiply)
- `distractor` -> `explanation`: about a number in the problem that the solution does not use
- `pushback` -> `explanation`: the pupil says the tutor is wrong and gives their own (wrong) answer; the tutor must hold firm
- `follow` -> `explanation`: a follow-up to the tutor's last explanation: still lost, why, say it simpler, an example, so does that mean...
- `what_if` -> `solution`: what the answer would be if a number were different (a new calculation: the tutor solves again)
- `off_topic` -> `out_of_scope`: not about this maths at all (other subjects, personal questions, jokes, games, rude); the harness answers

## Focus types and their slots

### `step`
- `{step_words}`: what the line computes, in the solution's words: 'number of bags'
- `{calculation}`: the calculation as written: '63 / 7'
- `{result}`: its result: 9
- `{ordinal}`: its position: 'first', 'second', 'third', 'fourth'
- `{a}`: first number of a one-operation step: 63
- `{b}`: second number: 7
- `{sign}`: the sign: '+', '-', '*' or '/'
- `{verb}`: 'add', 'take away', 'multiply' or 'divide'
- `{verb_past}`: 'added', 'took away', 'multiplied' or 'divided'
- `{sign_word}`: 'plus', 'minus', 'times' or 'divided by'
- `{operation_phrase}`: the operation with its numbers: 'divide 63 by 7', 'take 9 away from 54', 'add 54 and 9'
- `{operation_phrase_past}`: 'divided 63 by 7', 'took 9 away from 54', 'added 54 and 9'
- `{other_verb}`: an operation that would be wrong here: 'multiply'
- `{other_phrase}`: the wrong operation with the same numbers: 'multiply 63 by 7'
- `{a_words}`: the words for a, from the solution: 'new total'
- `{b_words}`: the words for b: 'bottles per bag'
- `{quotient}`: a remainder step's whole part: 5
- `{remainder}`: what is left over: 1

### `fact`
- `{number}`: the number: 7
- `{words}`: what it is, in the solution's words: 'bottles per bag'

### `answer`
- `{answer}`: the final answer: 9
- `{answer_words}`: what the answer is: 'number of bags'
- `{step_count}`: how many calculation lines: 2

### `distractor`
- `{distractor}`: the unused number: 4
- `{distractor_sentence}`: the sentence of the problem it is in: 'Hayrides cost $4 each.'

### `mistake`
- `{wrong_answer}`: the pupil's wrong answer: 45
- `{answer}`: the tutor's answer: 63
- `{wrong_verb}`: (wrong_operation) the operation the pupil used: 'take away'
- `{wrong_phrase}`: (wrong_operation) it with the numbers: 'take 9 away from 54'
- `{distractor}`: (used_distractor) the unused number the pupil used: 4
- `{skipped_words}`: (skipped_step) the step the pupil left out: 'number of bags'

### `what_if`
- `{old}`: the number in this turn's task: 9
- `{new}`: the changed number: 12
- `{old_words}`: the solution's words for it: 'bottles Ines added'

### `none`
- (no slots)

Slots exist only when the solution offers them (e.g. `a`/`b` only on one-operation steps, `a_words` only when both numbers are named). A template is used only where all its slots exist.

## Features (for `requires`)

- `add`: a one-operation addition step
- `subtract`: a subtraction step
- `multiply`: a multiplication step
- `divide`: a division step
- `multi_operation`: a step with more than one operation
- `remainder`: a division that leaves a remainder
- `rounded_up`: a remainder rounded up (16 / 3 -> 6 boxes)
- `uses_previous_step`: the step uses the result of an earlier line
- `uses_earlier_answer`: the step uses the answer of an earlier task
- `first_step`: the first calculation
- `last_step`: the last calculation
- `only_step`: the only calculation
- `has_words`: the solution names both numbers of the step (a_words and b_words exist)
- `from_problem`: the number is written in the problem
- `from_earlier_answer`: the number is an earlier answer
- `implied_by_word`: the number comes from a word: a pair = 2, a week = 7, a dozen = 12
- `one_step`: the solution has one calculation
- `multi_step`: two or more calculations
- `wrong_operation`: the pupil used another operation
- `used_distractor`: the pupil used the unused number
- `skipped_step`: the pupil stopped one step early
- `slip`: a small arithmetic slip
- `follow_up_task`: this turn is a follow-up task, building on an earlier answer
- `has_distractor`: the problem has an unused number
- `money`: the story is about money
- `time`: the story is about time (minutes, hours, days, weeks)
- `join`: story type: something is added to an amount
- `separate`: story type: something is taken away
- `combine`: story type: two parts make a whole
- `compare`: story type: how many more / fewer
- `equal_groups`: story type: equal groups (multiplication)
- `sharing`: story type: sharing equally (division)
- `grouping`: story type: how many groups of a size (division, can leave a remainder)
- `times_as_many`: story type: times as many
- `rates`: story type: rates and prices
- `mixed`: story type: several different steps

Story-type and conversation features (`money`, `time`, `join` ... `mixed`, `follow_up_task`, `has_distractor`) are on every focus of that conversation.

## Examples
See `library/domain/math_tutor/questions/bank/seed.jsonl` (37 lines).
