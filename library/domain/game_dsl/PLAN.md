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
3. **Commands** (Claude Sonnet agents, `sonnet_commands.py`; decided 2026-11-07 after a 60-problem test: 60/60
   pass the command check vs Gemma's 53/60, and they say what the code does instead of reading the syntax back):
   each problem's command in a mix of styles: terse `#name` ~50%, loose plain English ~25%, detailed ~15%, empty
   (autocomplete, /insert only) ~10%. ~2k Sonnet tokens per command. Pilot: 300 (281 asked + 19 empty).
4. **Gemma** (rented card, bf16; redesign 2026-11-07): Gemma WRITES each block from prompt + short input + command,
   keeping its probabilities while it writes (no separate scoring pass); the app's compiler + checks drop the bad
   ones; the student learns Gemma's blocks. Also measured: how often Gemma's block equals the real one (AST, exact
   and with names / numbers ignored). The same prompt for the training blocks and the test reference answers.
5. **Students** (local 4070): output vocabulary, corpus, KL + CE at a few sizes; report compile rate, clean rate,
   match with the real block, all next to Gemma's own answers.

## Splits

Test = every problem from the 8 held-out human games (Snake, Sokoban, Pinball-lite, Brick Breaker, Tower Defense,
Reaction Test, Open World, Sundown Herd) + ~15% of the new games (random within each family). Watch the test error
per pass: the coding agent's was lowest at pass 2, then rose (16 base projects were too few); ~140 base games
here.

## Size and cost (estimates; a 1-hour price test first)

~140 train games x ~30 versions x ~15 problems = ~60,000 problems. Gemma: commands + scoring ~8-12 h on an
RTX PRO 6000 ($2.49/h), ~$20-30. Phase 1 = the same with ~5 versions per game, end to end, before scaling.

## Next (decided 2026-11-07, not started: the user's usage was at 87%)

Measured: the 54,331 train problems hold only 2,773 distinct block shapes (names, numbers, colours ignored;
median 7 repeats, a plain wall 1,750x); 45% of the test targets are shapes never seen in train. So, in order:
1. More games and shapes (the main lever for unseen games): a second round of agents, ~120 more games.
2. More task types: `/fix` at the cursor like /edit (the block under the cursor broken by a rule: missing brace,
   keyword in the wrong block, argument count, misspelling, random where it is fixed; the input adds the
   compiler's or checker's message; the target is the real block); richer /edit kinds in both directions
   (removals, movement swaps, retarget, wrap in a guard, add health, add effects).
3. More ways of asking: 2-3 commands per problem in different styles (cheap; phrasing robustness only).
4. 10 versions per game instead of 30, each shape capped at ~20 problems.
Then the price test (~1 h): commands, scoring with the full manual / reference_card.md / nothing, Gemma's own
answers on test problems from the short input.
Price test v2 (Gemma writes; ~30 min): reference_card.md / full manual / the manual read once and shared by the
batch (must give identical output) / a moving card (the user's idea, 2026-11-07: only the card lines the problem
needs, chosen from what the model sees, never from the real block); compile + clean rate and cost per arm.

## v1 as it shipped (2026-11-08)

Student game-agent-kd-m-s1 (9.85M parameters, KL 0.9 -> 0.1, 10 passes) on 39,274 train examples: Gemma's checked
blocks for 54,450 commands (Sonnet 9,394, gpt-oss-120b 40,882 on rented cards, 3,876 bare /insert). On the 1,323
test problems: 90.9% of its blocks compile and pass the app's checks, 62.5% have Gemma's block's shape, 28.2% its
exact tree; it writes valid code but follows the command only loosely ("twice as big" made it smaller). Shipped in
the app (qr_flutter web/game_agent/v1, ONNX on-device, downloaded on first use; export_onnx.py).

## v2: the user's notes for the better model (2026-11-08)

1. Many ways of asking: with or without the verb words ("create", "add", "make", none), different sentence
   structures, sometimes common spelling mistakes; more examples in all.
2. Blocks from the user's point of view, not only the game's own blocks cut out: "here is a game, what could I
   possibly want to add or change here?" (new features, tweaks, characters the game does not have yet). A more
   faithful and richer set than re-adding blocks the game already has.
3. Stress-test the common asks: "add sounds" (every sound name, every place a sound goes); new characters of
   different shapes.
4. A training unit just for colours and shapes: colour words to hex (pinkish, light red, dark teal, ...) and shape
   words to shapes and sizes, so "a pinkish, light red circle character" gives exactly that.
5. Naming, taught well: "create a hero" gives the new block the id "hero"; when "hero" already exists in the game,
   the next free "hero_2", "hero_3", ... (hero_n). The same for every new name the command implies.
6. Commands that name the language's own constructs ("add an OnValue", "add a Wave", "add a timer", "add an
   OnCollision with #wall"): the model should know each construct's template (where it goes, its arguments, a
   sensible default body). v1 never learned these: Sonnet's instructions and the gpt-oss filter
   (open_commands.reads_code) kept keyword names out of every command, so a share of v2's commands should use them.
7. Bigger asks: whole levels and groups of blocks, by description or by likeness ("create a level with lava and
   moving platforms", "create a level that is like level 2 but harder", "add three enemies that patrol the top").
   v1 never cut a Level block (problems.insert_candidates: entities, events, action lists and window items only),
   so this is a new task kind; its targets are long (many blocks), which may need the larger student.
8. Edits said the way people point at things: "this" for the block under the cursor, no #name, short and casual,
   in many variations: "edit this knight from red to blue", "change this from a square to a circle", "change the
   text to whatever", "change to "winners"" (a quoted literal), "add text that reflects victory" / "text here for
   victory" (the words left to the model). v1's commands nearly always named the block (#knight) and spelled the
   change out.

## v2's corpus: the draft (2026-11-09)

Asks from the user's point of view (asks.py, wording.py): a whole game version (nothing cut out), the cursor at a
place a user could type, an ask, and Gemma writes the block (there is no real one). The input is built exactly as
the app builds it (live.py), so training and use match (v1's problems matched it on 74% / 90%). Units by rule, each
with an "expect" checked at home next to the compiler (does the block do what was asked: the colour, the shape, the
id, the keyword, bigger, faster, the text...): colour_shape, character, naming, construct, sound, edit (casual,
"this"), text, level; plus open asks without a rule check (hand-written or written by a model). Every ask goes
through wording.casual: please, a question, no capital, typos.

Gemma's prompt for v2 is "card2": the card + conventions (the naming rule; "this" = the block under the cursor; a
text is a text Entity; a new Level is a whole level with its own ids; no ";"; typos). Two slips are mended by rule
instead of dropping the block: ";" between actions, and a new Entity's id renamed to the naming rule's.

First test (A40, vLLM, 499 asks; runs/game_v2/test = the first prompt, test_b = with all the conventions). Usable
= compiles, passes the checks and does what was asked:

| unit | first prompt | card2 |
|---|---|---|
| colour_shape | 35% | 55% |
| character | 25% | 52% |
| naming | 68% | 78% |
| construct | 75% | 65% |
| sound | 78% | 78% |
| edit (casual) | 68% | 72% |
| text | 12% | 82% |
| level | 25% | 35% |
| open, hand-written | 52% | 78% |
| open, written by Gemma | 55% | 55% |

Gemma as the writer of open asks: plausible ideas, but at /insert places half its asks change something else ("make
the goalie faster"). What Gemma misses most: "huge" / "tiny", dark and light shades, levels that copy another
level's ids.

## Files

language.py (parser, checks, hollow), compiler.py (the app's compiler: qr_flutter/build/dsl_check/dsl_check.exe,
from qr_flutter/tool/dsl_check.dart), games.py (game files, split), manual.py (manual.md, reference.json),
check.py (the checker the agents use).
