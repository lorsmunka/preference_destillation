# Writing runs — the exact instructions every agent got

Every agent first read `../writer_brief.md`, then its assignment. The workflow scripts here
generated the assignments (one prompt per batch: slice, mix, contexts, style, seed lines or
reword lines, file name). Numbers per run are in `../items/writing_log.json`.

| run | script | model | what |
|---|---|---|---|
| pilot | (below) | Claude Sonnet 5.5 | 3 agents × 50 items: S04 memory, S04 online, seeded |
| A | math-tutor-write-a.js | Claude Opus 5.5 | 4 harvesters + 80 memory writers |
| B | math-tutor-write-b.js | Claude Opus 5.5 | seeded writers, 2 passes over the 1,310 original seeds |
| C | math-tutor-write-c.js | Claude Opus 5.5 | the 72 tasks A/B lost to the usage limit |
| D | math-tutor-write-d.js | Claude Opus 5.5 | last memory/seeded tasks + 2 passes over the 843 harvested seeds |
| reword 1 | math-tutor-reword.js | Claude Opus 5.5 | 2 rewordings per train item, chunks of 150 |
| reword 2 | math-tutor-reword-2.js | Claude Opus 5.5 | the chunks reword 1 lost to the usage limit |

Runs C and D re-issued the same prompts as A and B for the tasks that were missing; the seed
line offset in B's prompt was corrected in C/D (writers in B had checked their lines and
covered exactly the right seeds anyway).

## Pilot assignments (2026-10-04, Sonnet 5.5, launched directly)

All three: read the brief, write 50 items to a new file, run
`python -m library.domain.math_tutor.items <file>`, fix every failure, touch no other file,
no git; report items written, valid, and anything unclear.

**S04 memory** (`S04_memory_r01.jsonl`, ids S04-mem-r01-0001..0050): source memory; slice
S04; unknown difference 20 / larger 15 / smaller 15; grade 2 for 25, grade 3 for 25;
distractor none 25 / number 10 / sentence 7 / keyword 8; follow-ups 0 on 10, 1 on 18, 2 on
15, 3 on 7; at least 15 settings; comparison language in many forms; 8 real compare
problems from the seeds as style references (stories not to be reused).

**S04 online** (`S04_online_r01.jsonl`, ids S04-onl-r01-0001..0050): the same mix; find real
grade 2-4 additive comparison problems with WebSearch / WebFetch on teacher sites,
worksheets and open curricula, respecting the brief's blocklist; write adapted problems
(never copies) with `"source": "online:<url>"`; at least 8 source pages. Outcome: 1 of 11
pages gave usable text → online writers were replaced by harvesters (run A).

**Seeded** (`seeded_r01.jsonl`, ids sd-r01-0001..0050): 50 random seeds (seed 0); one new
item per seed with the same operations in the same order, a different story, setting,
names, wording and numbers; slice, unknown, grade, distractor and context chosen to fit;
follow-ups 0 on 10, 1 on 18, 2 on 15, 3 on 7; keep one irrelevant number where the seed
has one, else mostly none with about 5 sentence distractors.
