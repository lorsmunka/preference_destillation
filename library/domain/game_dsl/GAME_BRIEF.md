The brief for the agents that write new games (round 1, 2026-11-07: 8 agents x ~15 games, games/new/<family>/).
Fill the {placeholders}; one agent per family. Model: Sonnet (2026-11-07 test: 3 games, all passed the checker on
the first run, block shapes as new as round 1's; it wrote the largest one with a Python script, hence "by hand").

---

You are writing training data: about {count} NEW, ORIGINAL games in a small game language (the language of the user's app "QR Game Studio"). The games will be cut into pieces to teach a small model to write and edit this language, so every game must be correct, varied and actually playable, with every entity doing something meaningful.

## Your family: "{family}"
{description} Keywords to use often across your games (each at least twice in total): {keywords}. Vary controls, window sizes and orientations, themes, colours and music tracks.

## Read first (working directory D:\preference_destillation)
1. The full manual: library/domain/game_dsl/manual.md (the app's own docs, engine notes, a placement table at the end saying which keyword goes inside which block, sound and music names). Read ALL of it before writing.
2. A few of the user's own games as style references, ONLY from this list (library/domain/game_dsl/games/human/<slug>.dsl): meteor_run, bubble_storm, star_forager, space_invaders, boss_duel, long_run, platformer, last_light, moonward, plinko, pac_man, grid_run, grid_dungeon, dungeon_escape, lucky_slots, minesweeper, tetris, 2048. Do NOT open these held-out games (they are the test set): snake, sokoban, pinball_lite, brick_breaker, tower_defense, reaction_test, open_world, sundown_herd. Do not write clones of those eight either. Do not open library/domain/game_dsl/games/new/.

## Write
- Each game is one file: library/domain/game_dsl/games/new/{family}/<slug>.dsl (slug = short lowercase name with underscores). The file holds only the program (one Window block); # comments are allowed but never as placeholders for missing logic.
- Write every game by hand. No scripts that generate game code: generated code repeats itself, and the point is variety.
- After each game is clean, APPEND one line to library/domain/game_dsl/games/new/{family}/index.jsonl: {"slug": "...", "title": "...", "brief": "one or two sentences: what the player does and how they win or lose"}. Append only; never rewrite that file.
- Check every game: `python -m library.domain.game_dsl.check library/domain/game_dsl/games/new/{family}/<slug>.dsl` (run from D:\preference_destillation). It runs the app's real compiler, then checks what the compiler lets through (a keyword in the wrong block is silently ignored by the app, wrong argument count, bad colour, unknown id), then signs of a hollow game (empty blocks, a template never spawned, a signal nobody listens to or nothing sends, a bar never changed, no way to win or lose, fewer than 3 entities). It must print OK. Fix and re-check until it does; if a game still fails after ~5 attempts, simplify it or delete it.
- The checker cannot see gameplay logic, so reason it through yourself: positions are pixels from the bottom-left with y going up; size(8, 12) means 800 x 1200 px; startPosition is the bottom-left corner; collisions only fire between the ids you name (a bare OnCollision is the window border); spawned clones start where you spawn them; speeds in speed() are units of 100 px/s.

## Variety rules
- Each game a different core idea (no two with the same premise). Original premises and names only: no recreations of existing commercial games, their names or characters.
- Mixed sizes: ~1/5 small (40-80 lines), ~3/5 medium (80-200 lines), ~1/5 large (200-350 lines). Some with Level blocks (multi-level), most single-level.
- Mixed layout styles: some one-keyword-per-line, some compact (several keywords per line), like the user's games show.
- Mixed id styles (hero, p1, ship, rockA, dropper2, ...).
- Vary the code itself, not only the names: when a game needs several similar things, make them differ in behaviour (or use one template), rather than the same block pasted with another number.

## Do not
Touch any other file or folder, run git, or install anything.

## Report back (short)
A table: slug, lines, checker runs until OK, one-line premise. Then any place where the language did not behave as the manual says, or a rule you had to learn the hard way (one line each).
