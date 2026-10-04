# Writer brief — math tutor items

You write grade 2-4 maths word problems for training a small tutor model. Each problem
comes with a Python `solve()` function; the answer is whatever that function returns,
so the code must be exactly right. Problems are later checked by machine, deduplicated,
and some are given follow-up questions in a conversation, so the rules below matter.

## One item = one JSON line

```json
{"id": "S04-mem-r01-0001", "slice": "S04", "unknown": "difference", "grade": 2,
 "distractor": "none", "context": "library", "source": "memory",
 "problem": "Ana read 42 pages and Ben read 27 pages. How many more pages did Ana read than Ben?",
 "solve": "def solve():\n    ana = 42\n    ben = 27\n    more = ana - ben\n    return {'ana': ana, 'ben': ben, 'answer': more}",
 "follow_ups": [{"question": "How many pages did they read altogether?",
                 "solve": "def solve(state):\n    total = state['ana'] + state['ben']\n    return {**state, 'answer': total}"}]}
```

- `id`: the prefix you were given, then a 4-digit counter.
- `slice`, `unknown`: from the slice table below.
- `grade`: 2, 3 or 4. `context`: one word for the setting (bakery, farm, bus...).
- `distractor`: `none` | `number` (exactly one extra number, written in digits, that the solution does not use) |
  `sentence` (an extra sentence with no number) | `keyword` (wording that points to the
  wrong operation, e.g. "fewer" in a problem solved by adding).
- `source`: what you were told to put there.
- `follow_ups`: 0 to 3 questions that continue the same story.

## The code (checked by machine; anything else is rejected)

- `solve()` for the problem, `solve(state)` for each follow-up. `state` is the dict the
  previous turn returned.
- Only: assignments, `+ - * // %`, whole-number literals, `state['name']`, one final
  `return {...}` with an `'answer'` key. No function calls, loops, if, imports, floats.
- Every value along the way is a whole number from 0 to 999. No negatives.
- `//` must divide exactly. Only slice S07 may leave a remainder or use `%`.
  Round up as `(a + b - 1) // b`.
- Every number in the code must appear in that turn's text, as digits ("7") or a word
  ("seven", "twenty-one", "a dozen", "twice", "half", "a week"). 0 and 1 are always allowed.
- Numbers in the text are at most 99. Answers may go up to 999.
- Numbers written in digits that the code does not use must match the label:
  `distractor: number` needs exactly one, everything else needs none. Follow-ups never
  have unused digit numbers. Number words in ordinary prose ("the two classes", "three
  days") are fine; write every quantity the problem is about, and the distractor, in digits
  or in words the code uses.
- A follow-up must use `state[...]`. Store every quantity a follow-up needs in the
  returned dict (`return {**state, 'left': left, 'answer': left}` keeps earlier ones).
- One turn has 1 to 3 operations.

Check your file before you finish, and fix every failure:

    python -m library.domain.math_tutor.items <your file>

## Slices

| slice | what | unknown | grade |
|---|---|---|---|
| S01 | change, join: some more arrive | result, change, start | 2 |
| S02 | change, separate: some leave | result, change, start | 2 |
| S03 | combine: two parts, one whole | whole, part | 2 |
| S04 | compare, additive: more / fewer than | difference, larger, smaller | 2-3 |
| S05 | equal groups: n groups of k | product | 3 |
| S06 | partitive division: share equally | group size | 3 |
| S07 | quotative division: how many groups, leftovers, round up | group count, remainder, round up | 3-4 |
| S08 | multiplicative compare: 3 times as many, a third as many | larger, smaller, times | 3-4 |
| S09 | rates and money: price x quantity, change from a note, per hour | total, change, rate | 3-4 |
| S10 | multi-step: 2-3 operations mixing the above | result | 4 |

"Start unknown" means the beginning is missing: "Some birds sat on a fence. 6 more came
and now there are 15. How many were there at first?"

## Variety — the most important part

The model must learn to read, not to recognise templates. A problem that is another one
with new names or numbers is worthless (it will be deleted automatically).

- Write every problem individually. Vary sentence count (1-4), sentence order, where the
  question sits, how the question is asked. At least a third of your questions should not
  start with "How many" ("What is...", "Find...", "How much...", "How far...", "Work out...",
  "...how many does she have?", a question in the middle of the text).
- Vary the verbs and nouns of the story; use all of your assigned contexts.
- Names from many cultures, rarely repeated. Sometimes no name at all ("a teacher").
- Mix the unknown positions, grades, distractor styles and follow-up counts evenly
  over your batch, following the mix you were given.
- Plausible numbers for the story (no 87 friends at a table, no 4 pages in a book).
- No "Problem:" prefix, no answer, no hints, no equations in the text.
- Follow-ups continue the story naturally ("Then...", "The next day...", "If she gives
  2 to her sister..."), each with a new question that needs the previous answer or state.

## Your source

**memory** — write from your own knowledge. The seed problems you were given are real
problems for style reference only: do not reuse their stories.

**online** — search the web for real grade 2-4 word problems of your slice (worksheets,
textbooks, teacher sites, openly licensed curricula such as Illustrative Mathematics or
EngageNY / Eureka Math, maths blogs...). For each item, write an adapted problem: new
story details, new numbers, new wording — never a copy. Put `"source": "online:<url>"`
with the page that inspired it. Do NOT use: k5learning.com, commoncoresheets.com,
mathplayground.com, ck12.org, ixl.com, mathocean.com, ask-math.com, dadsworksheets.com,
khanacademy.org, math-aids.com, mathworksheets4kids.com, or any copy of the SVAMP, ASDiv,
MAWPS or GSM8K datasets (GitHub, Hugging Face, papers). Those feed our test set.

**seeded** — you get real problems (seeds) with their solve(). For each seed, write one
new problem with the same mathematical structure (same operations in the same order) but
a different story, setting, names and wording, and new numbers; choose the slice it fits.
Put `"source": "seeded:<seed id>"`. It must not read like the seed with words swapped.

## Writing the file

Create the file you were given (it must not exist yet; if it does, stop and say so) and
write one JSON object per line, UTF-8, no blank lines. Use `\n` inside the code strings.
When done, run the check, fix failures in your own file, run it again, and report:
items written, items valid, anything that was hard.
