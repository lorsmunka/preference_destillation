# The game-language agent: plan (2026-11-07)

A small student, trained from scratch, that works like a code assistant inside the QR Game Studio editor:
`/insert <command>` writes a new block at the cursor, `/edit <command>` rewrites the block under the cursor.
Commands name the game's parts with `#name` (an entity, a group of clones, a bar, a signal); the app checks every
`#name` before the model runs (one unknown name = the new block's own id). The answer is compiled and checked
before it is shown; on an error the harness asks once more with the compiler's message.

Proven on 2026-11-06 (Gemma 4 12B, bf16, 40 Entity blocks cut from the human games): with the manual Gemma writes
blocks that compile and pass every check 36/40 (without it 17/40); teacher-forced, its top guess is the real
token 97.6% (syntax) / 95.6% (numbers, colours, ids). Whole games compile (11/12) but are often hollow.

## What the model sees (no agentic step: the cursor gives the place, the #names give the parts)

```
WINDOW   size, background, gravity, world size
NAMES    one line per entity / group / bar / signal (kind, shape, movement, events)
CURSOR   where it is (top level between #a and #b / inside #x's Events / ...); for /edit the block itself
#name    the full code of every #name in the command
COMMAND  /insert ... or /edit ...
```
~600-1,200 tokens typical (the whole game: median 1,860, up to 14,600). Gemma sees the same plus the manual.

## Stages

1. **Base games** (Claude agents, 8 families x ~15 = ~120 new games; `games/new/<family>/`). Original premises,
   every game passes `python -m library.domain.game_dsl.check` (the app's compiler + placement / argument / colour /
   id checks + not hollow). Agents see the manual and the 18 train human games only.
2. **Problems** (scripts, no GPU): seeded versions of every game (ids renamed, colours, positions, numbers, entity
   order) x ~30; `/insert` cuts at every level (~40% Entity, 30% event, 30% action list, some Window items);
   `/edit` pairs from rule-based changes (speed, size, colour, shape, movement kind, an action added or removed,
   an event added), where the target is always the real block and the "before" is the changed one.
3. **Gemma** (rented card, bf16): (a) words each problem's command in a mix of styles: terse `#name` ~50%, loose
   plain English ~25%, detailed ~15%, empty (autocomplete, /insert only) ~10%; (b) scores the real block given
   the short input + command (teacher-forced) = the teacher data; (c) on the test problems also writes its own
   answer, the reference the student is compared with.
4. **Students** (local 4070): output vocabulary, corpus, KL + CE at a few sizes; report compile rate, clean rate,
   match with the real block, all next to Gemma's own answers.

## Splits

Test = every problem from the 8 held-out human games (Snake, Sokoban, Pinball-lite, Brick Breaker, Tower Defense,
Reaction Test, Open World, Sundown Herd) + ~15% of the new games (random within each family). Watch the test error
per pass: the coding agent's was lowest at pass 2, then rose (16 base projects were too few); ~140 base games
here.

## Size and cost (estimates; a 1-hour price test first)

~140 train games x ~30 versions x ~15 problems = ~60,000 problems. Gemma: commands + scoring ~8-12 h on an
RTX PRO 6000 ($2.49/h), ~$20-30. Phase 1 = the same with ~5 versions per game, end to end, before scaling.

## Files

language.py (parser, checks, hollow), compiler.py (the app's compiler: qr_flutter/build/dsl_check/dsl_check.exe,
from qr_flutter/tool/dsl_check.dart), games.py (game files, split), manual.py (manual.md, reference.json),
check.py (the checker the agents use).
