# The game language: a reference manual

Games are written in a small language. A program is one Window block. The only grammar rule:
`keyword(parameters) { child blocks }` -- both parts optional. Parentheses hold only strings, numbers,
true/false and random(min, max); everything nested goes in braces. Keywords are case-insensitive, but
write them as below. A keyword in the wrong block is silently ignored, so follow the placement table.


## Comments and whitespace

Anything on a line after a "#" is treated as a comment and ignored by the compiler. Indentation and line breaks are free-form -- only the nesting of parens (...) and brace blocks { ... } matters. Parens and braces are NOT interchangeable: a call's parentheses hold only its base-type parameters -- the strings, numbers, and booleans that are the leaves of the tree -- while every nested child node goes in a { } brace block. A call may have both: `OnTapped("btn") { Actions { ... } }` carries its id in the parens and its body in the braces.

```
# A throwaway comment.
Window {
    size(8, 8)   # inline comments work too
}
```

Parameters go in the parens, children go in the braces. OnTapped takes one string parameter ("btn") and one child node (Actions), which in turn holds the updateBar leaf.

```
OnTapped("btn") { Actions { updateBar(1) } }
```

A keyword that only takes parameters starts with a lower-case letter: size, rectangle, burst, sound. The blocks that give a program its shape start with a capital: Window, Entity, Type, Events, OnTapped, Phase, Actions. An action stays lower-case even when it holds a block: whenBar, whenVar, delay, randomAction. The compiler reads any letter case, so older code written as Burst(...) still runs, but a keyword it does not know is an error.

```
Entity {
    id("coin")
    Type { circle(20) }
    Events { OnTapped { Actions { burst(8, "fbbf24") } } }
}
```


## Window

Every program starts with a Window. It declares the playfield dimensions (in DSL units -- each unit is 100 px), the background color, the gravity that pulls AutoMove entities down, an optional background music track (see "Background music"), and any number of Bars. For larger-than-window worlds with a follow-camera, see "Camera and worldSize" next. Beside the editor on the Code tab (under it on a phone), the Size tool sets size(...) for you, Position turns a click in the window into the x, y to type, Color picks a color and gives its hex, and Sound plays the sounds and music and puts the one you pick into the code.

Minimal Window -- defaults to 900x900 px, black, no gravity.

```
Window
```

A typical portrait window with a sky-blue background, gravity and a music loop. Gravity is in px/s^2 (pulling down); the bundled Pinball, Plinko and Platformer games use 450-700. Background colour accepts the same formats as Entity color (with optional alpha).

```
Window {
    size(9, 16)
    background("87CEEB")
    gravity(600)
    music("bg_ambient")
}
```


## Camera and worldSize

By default the world IS the window -- entity positions are clamped to size(). To make a side-scroller or open-world game, declare a worldSize larger than size() and a cameraFollow target. The painter renders the whole world into a canvas and translates that canvas so the target stays centred (clamped to world bounds so the camera never reveals off-world space). Tap and pan coordinates are converted to world coords automatically, so tap zones remain clickable even when the camera has scrolled them far from their screen positions.

A 5-screen-wide side-scroller. The viewport is 800x900; the world is 4000x900. The runner is followed; the camera scrolls horizontally as the runner moves east.

```
Window {
    size(8, 9)
    worldSize(40, 9)
    cameraFollow("runner")
    background("0f172a")
}
```

Both-axis camera. World is 2x viewport in each direction; the camera tracks the hero across the whole 2x2 grid.

```
Window {
    size(8, 9)
    worldSize(16, 18)
    cameraFollow("hero")
}
```


## Units and coordinates

Positions and sizes are in pixels, counted from the bottom-left corner of the world with y going up; startPosition puts an entity's bottom-left corner there. The window and world sizes are the exception: they are in units of 100 px, so size(8, 12) is 800 x 1200 px. Speeds set with speed(...) are in units of 100 px per second, while setVelocity and applyImpulse take pixels per second. Times are in seconds. The Position tool beside the editor turns a click in the window into these coordinates.

Units of 100 px: the window, the world, and how far a chaser keeps back.

```
Window {
    size(8, 12)          # 800 x 1200 px
    worldSize(20, 30)    # 2000 x 3000 px
    gravity(900)         # px per second, per second
}
Movement { speed(6) chase("hero", 0.75) }   # 75 px behind
```

Pixels: positions, shapes, paths and grid steps.

```
startPosition(120, 40)      # bottom-left corner, y up
Type { rectangle(60, 80) }  # 60 x 80 px
Movement { GridMove(100, 0.3) { direction(0, 0) } }  # 100 px steps
```

Speeds: speed and updateSpeed are in units of 100 px/s; setVelocity and applyImpulse in px/s. These two move at the same 170 px/s.

```
Movement { speed(1.7) AutoMove { direction(1, 0) } }
Actions { setVelocity(170, 0) }
```


## Entity

An Entity is anything that exists on screen. It has an id, an initial position (in pixels), a color, an optional per-entity gravity override, a Type, an optional Movement, and any number of Events that react to taps, collisions, or elapsed time. Every entity needs a unique id. startPosition is the sprite's bottom-left corner, not its centre.

A green rectangle "ball" at (200, 300). startPosition is the bottom-left corner, so this 40x40 ball spans x 200-240, y 300-340.

```
Entity {
    id("ball")
    startPosition(200, 300)
    color("4ade80")
    Type { rectangle(40, 40) }
}
```

Per-entity gravity. Without this, the entity inherits whatever gravity the Window declared. Setting it to 0 keeps an entity weightless in a gravity world; setting it higher than the window's makes that one entity fall faster.

```
Entity {
    id("ball")
    startPosition(400, 600)
    color("fbbf24")
    gravity(0)
    Type { circle(20) }
    Movement { speed(4) AutoMove { direction(1, 0) } }
}
```

Color accepts "RRGGBB" or "#RRGGBB" (alpha implicit FF) or "AARRGGBB" / "#AARRGGBB" with an explicit alpha. The short forms "RGB" and "ARGB" (with or without #) double each digit, so "f80" is "ff8800". "00000000" is the fully-transparent marker used for invisible controller entities (spawners, scorekeepers).

```
color("4ade80")           # opaque green
color("#4ade80")          # same, # optional
color("000000")           # opaque black
color("fff")              # opaque white, short form
color("80ffffff")         # half-alpha white
color("8fff")             # same, short form
color("00000000")         # fully transparent
```

Three boolean flags on an Entity, each opt-in: solid(true) blocks Control / chase entities (and grid steppers) from passing through; oneWay(true) is a Mario-style platform that catches a descending gravity-affected entity from above but lets it jump up through; pushable(true) is a Sokoban box -- a stepping GridMove entity shoves it one cell if the cell behind is free.

```
Entity { id("wall") startPosition(0, 0)
    color("1e293b") Type { rectangle(800, 20) }
    solid(true) }
Entity { id("ledge") startPosition(60, 280)
    color("f97316") Type { rectangle(220, 16) }
    oneWay(true) }
Entity { id("crate") startPosition(350, 800)
    color("a16207") Type { rectangle(100, 100) }
    pushable(true) }
```

facing("left") or facing("right") says which way the drawing faces. While the entity moves the other way the engine draws it mirrored, and it keeps facing where it last went when it stops. Emoji animals mostly face left, so a horse emoji with facing("left") gallops head first both ways.

```
Entity { id("hero") startPosition(400, 600)
    color("ffffff") Type { text("🐎", 58) }
    facing("left") }
```


### Shape types

Ten shape primitives, all axis-aligned bounding box (AABB) for collision. The polygon-family shapes (polygon, star, ring) have square bounding boxes 2*radius wide; diamond and cross take explicit width and height. tetromino and the joystick are documented in their own sections; text has one too.

Three classic primitives. triangle's second arg picks which vertex is the point ("up", "down", "left", "right").

```
Type { rectangle(80, 30) }
Type { circle(20) }
Type { triangle(50, "up") }
```

Regular n-gons and stars are parameterised. polygon(sides, radius) covers hexagons (6), pentagons (5), octagons (8). star(points, outerRadius, innerRadius) has 2*points vertices alternating between the two radii; innerRadius defaults to outerRadius*0.45 if omitted.

```
Type { polygon(6, 30) }           # hexagon
Type { polygon(5, 30) }           # pentagon
Type { star(5, 30, 14) }          # 5-point star
Type { star(6, 24) }              # 6-point, default inner
```

ring is an annulus (filled circle with a hole); innerRadius defaults to outerRadius*0.6. diamond and cross take a rectangular bounding box -- cross's third arg is the bar thickness, default 30% of sizeX.

```
Type { ring(28, 14) }             # donut
Type { diamond(60, 60) }          # rhombus
Type { cross(40, 40, 12) }        # plus sign
```


## Movement

Entities are static by default. Add a Movement to make them move. Five flavours: AutoMove, Control, Path, chase, GridMove. AutoMove and Control are below; the others get their own sections after this one.

AutoMove: pure velocity. direction is a multiplier on speed, not normalised — (1, 0) is rightward, (0, -1) is down, (1, -1) is down-right, and (2, 0) moves twice as fast as (1, 0) -- until an updateSpeed, which sets the speed itself and keeps only the heading. Gravity from the Window pulls AutoMove entities down each frame.

```
Movement {
    speed(3)
    AutoMove { direction(1, -1) }
}
```

Control: the entity follows the player's drag finger. movementControl gates which axes are draggable -- 0 = both, 1 = vertical only, 2 = horizontal only. With movementControl(2) AND non-zero window gravity, the entity is a platformer hero: drag for sideways input, gravity pulls it down. Other axis modes are gravity-immune (drag wins).

```
Movement {
    speed(5)
    Control { movementControl(2) }
}
```


## Path movement

Path: designer-authored motion. The entity walks its waypoints in declared order at speed(N), then loops back to the first after the last -- two waypoints make a back-and-forth patrol, three+ make a circuit. Path ignores gravity and the border clamp (it owns its position), and starts on the first waypoint regardless of startPosition. Add dwell(N) to pause N seconds at every waypoint -- useful for giving the player a timing window to slip past a guard.

A guard that paces left-right forever. Swap in more waypoints for a longer route.

```
Movement {
    speed(3)
    Path {
        waypoint(90, 900)
        waypoint(700, 900)
    }
}
```

Same patrol, but with a 1s pause at each endpoint. The dwell() child is a sibling of waypoint(...).

```
Movement {
    speed(3)
    Path {
        waypoint(90, 900)
        waypoint(700, 900)
        dwell(1)
    }
}
```


## Chase movement

chase("targetId") is the fourth Movement flavour: the entity homes in on another entity every frame, always closing the gap at speed(N). It is the simplest enemy AI -- a guard that hunts the player, or a homing shot. A second number keeps a distance: chase("hero", 0.7) stops 0.7 units short of the hero, so it trails behind instead of sitting on it. Like Path it owns its position (gravity does not apply) and is stopped by solid entities. The retarget("newId") action repoints a chaser at a different target mid-game.

A guard that chases the entity with id "player" for the whole game.

```
Movement {
    speed(3)
    chase("player")
}
```

A follower keeps 0.7 units behind the hero; a second follows the first, and the two trail along like a rope.

```
Movement { speed(6) chase("hero", 0.7) }
Movement { speed(6) chase("first", 0.6) }
```

retarget swaps the target: a homing shot chases the player for 6s, then turns on the boss.

```
Events {
    OnElapsedTime(6.0) { Actions { retarget("boss") } }
}
```


## Grid movement

GridMove is discrete motion -- the entity jumps a fixed step every fixed interval instead of flowing at a velocity. GridMove(stepPixels, secondsPerStep) { direction(dx, dy) } sets the jump size, the cadence, and the starting heading; a step is cancelled if it would land in a solid entity or leave the world. The setDirection("id", dx, dy) action re-aims a GridMove entity at runtime -- wire it to tap-pads to steer. Grid mode coexists with the arcade movements: a game can mix a GridMove avatar with Path or AutoMove entities freely. An OnGridStep event fires once on each actual step -- bind per-step behaviour to it (leaving a trail, broadcasting) so it runs on the movement own clock, never a separate timer that can drift out of sync. Its counterpart OnGridBlocked fires when a step is stopped by a wall or the border -- wire it to gameOver to make walls lethal.

A runner that steps one 100px cell every 0.3s, and a pad that steers it up.

```
Entity {
    id("runner")
    startPosition(60, 440)
    color("22d3ee")
    Type { circle(40) }
    Movement { GridMove(100, 0.3){ direction(1, 0) } }
}
Entity {
    id("padUp")
    startPosition(335, 170)
    color("94a3b8")
    Type { rectangle(130, 130) }
    Events {
        OnTapped("padUp") { Actions { setDirection("runner", 0, 1) } }
    }
}
```

OnGridStep fires once per step. Here the head drops a trail segment in each cell it leaves -- one drop per move, so the trail can never drift out of sync.

```
Events {
    OnGridStep {
        Actions { spawnHere("seg") }
    }
}
```

gridStep(dx, dy) shifts a GridMove entity instantly by one cell in (dx, dy) -- no time delay, full collision + push semantics. With an optional leading id arg, it targets a named entity. Right primitive for Sokoban-style "tap to step once".

```
Actions { gridStep(0, -1) }            # step source down
Actions { gridStep("hero", -1, 0) }    # step hero left
```


## Pushable solids

pushable(true) on an Entity turns it into a Sokoban box. When a GridMove entity tries to step into a pushable, the engine moves both in lockstep IF the cell behind the pushable is free. Two pushables in line cancel; a pushable + a non-pushable solid in the same step cancels. Pushable is treated as solid for collision targeting too -- the stepping entity cannot pass through.

A box. Walk the hero into it; the box shoves one cell as long as the cell behind is empty.

```
Entity { id("crate") startPosition(350, 800)
    color("a16207") Type { rectangle(100, 100) }
    pushable(true) }
```


## One-way platforms

oneWay(true) on an Entity makes it a Mario / Doodle-Jump-style platform. Each frame, the engine checks every gravity-affected entity (AutoMove or horizontal-only Control with window gravity) that was ABOVE the platform last frame and is now descending into it -- those are snapped to the platform top with velocityY zeroed. Going up through a one-way platform passes through cleanly; sideways motion ignores it.

A ledge in a platformer. Tap-pads + setVelocity (see Physics actions) jump the hero up through this ledge, then drop them back onto it.

```
Entity { id("ledge") startPosition(60, 280)
    color("f97316") Type { rectangle(220, 16) }
    oneWay(true) }
```


## Events

Events sit inside an Entity's Events { } block and fire when something happens. Each event holds one Actions { } block, run once per fire.

OnCollision("X") fires when this entity overlaps entity X. With no argument, it fires when this entity hits the window border.

```
Events {
    OnCollision("paddle") { Actions { bounceBack() } }
    OnCollision { Actions { bounceBack() } }
}
```

OnTapped("X") fires when the user taps THIS entity AND its id is "X" -- effectively a self-tap guard. The OnTapped ("X") form does NOT cross entities: declaring `OnTapped("button")` inside a guard's Events block does nothing when "button" is tapped, because the engine only fires the handler whose declaring entity's id matches the arg. Use OnTapped { ... } (no arg) to catch every tap on this entity regardless of id.

```
Events {
    # Fires when "button" itself is tapped.
    OnTapped("button") { Actions { updateBar(1) } }
    # Fires on ANY tap on this entity (id-agnostic).
    OnTapped { Actions { setText("prompt", "Nice!") } }
}
```

OnElapsedTime(N) fires once when N seconds have passed since this entity first existed; with no time, OnElapsedTime { ... } fires once as the entity appears (the usual place to set it up). For entities present at level start that's the same as game start; for SPAWNED clones, the clock begins at spawn time (a 3s timer on a fresh meteor fires 3s after THAT meteor appeared, not 3s after the level loaded).

```
Events {
    OnElapsedTime(3.0) { Actions { gameOver("time up!") } }
}
```

Add repeatCount(K) inside the OnElapsedTime body to re-fire on the same cadence. K = a positive number fires K times then stops; K = -1 fires forever. Default (no repeatCount) = 1 = the one-shot above.

```
Events {
    # Fires at 1s, 2s, 3s — three pulses total.
    OnElapsedTime(1.0){ repeatCount(3) Actions { updateBar(1) } }
    # Fires every 0.5s forever.
    OnElapsedTime(0.5){ repeatCount(-1) Actions { updateBar(1) } }
}
```

OnSwipe("dir"){ Actions { ... } } fires when the player swipes in the given direction ("up", "down", "left", "right"). A swipe is recognised at PAN END (finger lift): if the total displacement exceeds 40 px and the pan did not grab a Control entity, the dominant axis becomes the direction. OnSwipe with no direction arg catches every swipe. The 2048 template puts these handlers on an otherwise invisible entity that owns all four directional broadcasts.

```
Events {
    OnSwipe("left") { Actions { broadcast("slideLeft") } }
    OnSwipe("right") { Actions { broadcast("slideRight") } }
    OnSwipe { Actions { updateBar("any", 1) } }
}
```

OnGridStep / OnGridBlocked are for GridMove entities. They fire on a real step / on a step blocked by a wall or border. See the Grid movement section.

```
Events {
    OnGridStep { Actions { spawnHere("seg") } }
    OnGridBlocked { Actions { gameOver("WALL") } }
}
```

OnSignal("name") fires when any other entity broadcasts the matching signal name. See the Signals section.

```
Events {
    OnSignal("door-opened") { Actions { setColor("4ade80") } }
}
```


## Actions

Actions are what an Event's body does when it fires. They mutate the running game state -- change colors, teleport entities, bounce, delete, win, lose. Multiple actions in one block fire in order.

IMPORTANT: many actions default to "source" -- the entity whose Events { } block fired this action. Calling bounceBack() inside guard1's OnCollision bounces guard1; calling deleteEntity() inside ball's OnCollision deletes the ball. setPosition, setText and setDirection always name the entity they move or change. setVelocity, applyImpulse, gridStep, setScale, setOpacity, setVisible, setEnable, deleteEntity, flash and burst act on the source unless an id comes first -- setVelocity("hero", 0, 520), deleteEntity("brick3"), flash("hero", "ff0000", 0.2); the aimed form needs all its arguments.

How the id is matched: setVelocity, applyImpulse, setScale, setOpacity, setVisible and setEnable also match a TEMPLATE id and then act on every clone of it (every ball in a multi-ball game); gridStep takes the first match. setPosition, setText, flash and burst look for that exact id and fall back to the source when there is none; deleteEntity and setDirection then do nothing.

The action keywords covered here are the everyday ones; see the dedicated sections for delay, broadcast, whenBar/whenVar, randomAction, and the spawning family.

Position and motion: teleport, slide, bounce, change speed, retarget a chaser.

```
Actions {
    setPosition("ball", 100, 500, 0)    # teleport
    setPosition("ball", 100, 500, 2)    # slide over 2s
    setDirection("hero", 1, 0)          # GridMove dir
    bounceBack()                        # reflect velocity (paddle-style spin)
    bounceBack(0)                       # pure mirror, no spin
    updateSpeed(7)                      # source.speed = 7
    retarget("boss")                    # repoint a chase entity
}
```

Physics: velocity primitives for pinball flippers, platformer jumps, aim-and-fire slingshots. See "Physics actions" below for the full story.

```
Actions {
    applyImpulse(0, 300)             # ADD to source's velocityY
    applyImpulse("ball", 220, 480)   # to a named target
    setVelocity("hero", 0, 520)      # REPLACE (no stacking)
}
```

Display: color, text, scale, opacity, visibility, full-disable. setEnable(false) parks an entity entirely (no render, no update, no collisions, no events).

```
Actions {
    setColor("FF0000")
    setText("prompt", "GO!")
    setScale(0.5)                    # half size on source
    setScale("ball", 1.5)            # named target
    setOpacity(0.3)                  # fade source
    setVisible(false)                # hide source (still ticks)
    setEnable(false)                 # park source completely
}
```

Scoring + life cycle: bars, vars, game-over, level advance, restart.

```
Actions {
    updateBar(1)                     # default bar +1
    updateBar("score", 5)            # named bar
    changeVar("hp", -1)              # per-entity var
    gameOver("YOU WIN")
    nextLevel()                      # advance multi-level game
    restart()                        # reset current level
}
```

Removal: deleteEntity removes the source (or a named entity); deleteCollider removes the entity in the source's most recent collision (used by Space Invaders' bullet to drop the specific invader it hit).

```
Actions {
    deleteEntity()                   # source
    deleteEntity("brick3")           # by id
    deleteCollider()                 # the other in source's last collision
}
```

Sliding tiles (puzzle / merge games). slide(dx, dy) walks the source tile one cell at a time toward (dx, dy) until blocked; with a mergeVar arg, two tiles sharing the var's value merge (delete blocker, double the survivor's var). slideAll is the global version: it sorts every entity of a templateId by direction-aware distance to the wall (closest first), slides each, and locks a freshly-merged tile from merging again that turn -- the per-turn cap real 2048 requires. Use slideAll for swipe-the-whole-board flows; slide for one-tile pushes.

```
OnSwipe("left") { Actions {
    slideAll(-1, 0, 130, "tile", "value", "recolor")
} }

OnSignal("nudge") { Actions { slide(-1, 0, "value", "recolor") } }
```

loopCount(N) repeats every action after it in the block N times per fire, until the next loopCount. loopCount(-1) is the same as loopCount(1): one run per fire, with no re-firing while the event condition holds.

```
Actions {
    loopCount(3)
    updateBar(1)        # +3 per fire
}
```


## Physics actions

Two velocity primitives, both with the (optional id, dx, dy) signature: applyImpulse ADDS to existing velocity (good for stacking kicks), setVelocity REPLACES (good for "jump"-style fixed boosts where holding the button can't rocket the entity). Both cancel an in-flight setPosition slide so the velocity actually takes effect.

Pinball flipper: applyImpulse stacks on the falling ball's velocity, so a well-timed tap reverses direction without snapping. With three live balls (template spawn), templateId match fans the impulse across all of them.

```
OnTapped("flipL") {
    Actions { applyImpulse("ball", 220, 480) }
}
```

Platformer jump: setVelocity is the right choice -- pinning vy at 520 means spamming the button doesn't accumulate into a rocket and skip past platforms.

```
OnTapped("padL") {
    Actions { setVelocity("hero", -180, 520) }
}
```


## Display setters

Four actions for tweaking how an entity renders, all sharing the (optional id, value) signature: setScale, setOpacity, setVisible, setEnable. setScale multiplies BOTH visual size AND hitbox; setOpacity multiplies into the render alpha; setVisible toggles the render skip flag (entity still ticks); setEnable freezes the entity completely.

Charging power-up: scale up over time. The hitbox grows with the visual.

```
Actions {
    setScale("hero", 1.5)
    setOpacity("hero", 0.6)
}
```

Reveal a hidden door: it was disabled at level start (no render, no collisions); a key broadcast wakes it up.

```
Entity { id("hidden") startPosition(400, 600)
    color("a855f7") Type { rectangle(80, 80) }
    Events {
        OnSignal("found") { Actions { setEnable(true) } }
    } }
# elsewhere, at level start:
# OnElapsedTime(0.05){ repeatCount(1) Actions { setEnable("hidden", false) } }
```


## FX primitives

Three short, no-gameplay-effect visual flourishes: flash tints an entity for N seconds, shake jitters the whole canvas, burst sprays N colored particles from the source. Every polished template layers these on top of its core actions -- a hit becomes a flash + burst, a win adds a shake.

flash: tint the source (or a named entity) with a color for N seconds. The original color returns when the flash expires.

```
Actions {
    flash("ff0000", 0.25)            # flash source red
    flash("hero", "ff0000", 0.25)    # flash "hero" red
}
```

shake the whole canvas (intensity, seconds). Layered after a collision to sell the impact. A new shake replaces an in-progress one; they don't stack.

```
Actions { shake(6, 0.35) }
```

burst N short-lived particles in the given color from the source's position. Particles tween outward and fade.

```
Actions { burst(14, "ef4444") }
```


## Sound effects

sound("name") plays one of the sound effects listed below. It is an action like any other, so it goes wherever the moment happens: next to the updateBar that scores a point, the gameOver that ends the run, the burst that breaks a brick. Several sounds can play at once; the same sound fired again within a few milliseconds plays only once, so ten bricks breaking together sound like one break.

The list is every audio file in the app's Sound_effects folder. To add a sound, put an MP3 (or WAV, OGG, M4A) into qr_flutter/Sound_effects/ and rebuild or restart the app: it shows up here, and sound("its file name without the extension") plays it. A file whose name starts with bg_ is background music instead (see Background music). Every sound is played at the same loudness, so a file recorded quietly or loudly needs no editing. The web app measures each file when it loads; the native apps use the levels that dart run tool/measure_sounds.dart measures, so run that after adding a file.

The audio ships inside the app, and the QR code only carries the name, so a sound costs a few bytes -- but a game that uses a sound you added only plays in a build that has the file. A name that is not in the list is a compile error. Browsers only start audio after a tap or click, which pressing Play counts as.

Sound names: beep, bell, blip, bubble, bump, buzz, clang, click, coin, death, glass, hurt, jump, laser, lose, pickup, pop, sparkle, thud, win, win2

Layer a sound on the moments that matter: collecting, getting hit, winning, losing.

```
OnCollision("coin") {
    Actions {
        sound("coin")
        updateBar(1)
        burst(8, "facc15")
    }
}
OnCollision("enemy") {
    Actions {
        sound("hurt")
        shake(6, 0.3)
        updateBar("lives", -1)
    }
}
# win and lose
Actions { sound("win") gameOver("YOU WIN!") }
Actions { sound("lose") gameOver("GAME OVER") }
```


## Background music

music("name") in the Window loops one of the tracks below for the whole game. It keeps playing through restart() and from level to level, and stops when the game ends, so the win or lose sound is heard on its own. Leave it out for a silent backdrop; sound effects play either way.

The tracks are the files in Sound_effects/ whose name starts with bg_: drop in bg_anything.mp3, rebuild, and music("bg_anything") loops it. Like sound effects, the tracks ship inside the app and the QR code carries only the name. Every track is levelled to the same loudness, 3 dB under the sound effects, so switching games never jumps in volume and the music never covers the effects. An unknown name is a compile error.

Music tracks: bg_1, bg_2, bg_3, bg_4, bg_5, bg_6, bg_ambient, bg_nature

Pick one track per game, next to the other Window settings.

```
Window {
    size(8, 12)
    background("0a0e27")
    music("bg_3")
    # ...entities, bars, events
}
```


## Difficulty primitives

Three structural keywords for shaping how a game ramps. Wave declares timed spawn waves inside Window; ramp lerps a Bar from one value to another over time; the restart action resets the current level without ending the game. The Sequence event (see Sequence) is the per-entity sibling for phased behaviour.

Wave: each Wave child of Window is one timed firing band on a synthetic director entity. Wave(durationSec, intervalSec){ Actions { ... } } fires its actions every intervalSec for durationSec total. Skipping intervalSec defaults to durationSec (one fire at wave end).

```
Window {
    size(8, 12)
    # Three spawn waves of escalating tempo.
    Wave(6, 1.3){ Actions { spawn("meteor", random(60, 720), 1180) } }
    Wave(9, 1.0){ Actions { spawn("meteor", random(60, 720), 1180) } }
    Wave(12, 0.7){ Actions { spawn("meteor", random(60, 720), 1180) } }
}
```

ramp lerps a bar from one int to another over N seconds. The bar passes through every intermediate integer, so OnValue triggers on that bar fire in sequence -- great for a "difficulty climbs from 1 to 10" curve.

```
Actions { ramp("difficulty", 1, 10, 30) }
```

restart() resets the current level from scratch (entities to startPositions, bars back to what they were when the level began -- their initials on the first level), no game-over overlay. Used in brick_breaker when the ball drains -- the player retries the same stage in place.

```
OnCollision("ball") {
    Actions {
        sound("hurt")
        shake(6, 0.35)
        burst(16, "FF4500")
        restart()
    }
}
```


## delay

delay(seconds){ Actions { ... } } schedules an action block to run later. The rest of the current block continues immediately -- delay defers, it does not pause. It is how you build a telegraph: flash a warning now, strike a beat later. delays nest and chain freely for multi-step sequences.

A telegraphed attack: the entity flushes red now, and 0.7s later spawns a hazard and reverts its colour.

```
Actions {
    setColor("ef4444")
    delay(0.7){ Actions {
        spawn("hazard", 400, 980)
        setColor("a855f7")
    } }
}
```


## Sequence

Sequence is a self-driving event -- it sits in Events { ... } and runs from the moment the entity exists. It holds an ordered list of phases. Each Phase(repeatCount, everySeconds){ Actions { ... } } fires its actions repeatCount times, everySeconds apart, then hands control to the next phase. A repeatCount of -1 is a terminal phase that runs forever. Use it for phased behaviour -- a boss that opens with one attack pattern, then escalates to a harder one.

A boss attack that escalates: three measured volleys, then a faster pattern forever.

```
Events {
    Sequence {
        Phase(3, 2.8){ Actions { spawn("shot", 400, 980) } }
        Phase(-1, 1.6){ Actions { spawn("shot", 400, 980) } }
    }
}
```


## Signals

broadcast("name") fires a named signal, and every OnSignal("name") event on any entity runs in response. It is the decoupled counterpart to a collision -- one entity triggers another without touching it. Use it for game-wide moments: a switch that opens a distant door, an enemy that enrages a whole room at once.

A key, when tapped, broadcasts "unlock"; a door elsewhere listens for it and removes itself.

```
Entity {
    id("key")
    startPosition(100, 100)
    color("fbbf24")
    Type { circle(20) }
    Events {
        OnTapped("key") { Actions { broadcast("unlock") } }
    }
}
Entity {
    id("door")
    startPosition(700, 100)
    color("78716c")
    Type { rectangle(40, 120) }
    Events {
        OnSignal("unlock") { Actions { deleteEntity() } }
    }
}
```


## Random values

random(min, max) replaces a literal number and is re-rolled each time it is used. It works in any number an action takes (spawn, setPosition, updateSpeed, updateBar, gridStep, ...) except the value whenBar and whenVar compare against, and as the time of an OnElapsedTime or a Phase — random(1, 3) waits a fresh 1-3s before every fire (the unpredictable-delay pattern). Using random(...) for a value fixed when the game is built — window size, startPosition, shape size, movement speed, or OnElapsedTime repeatCount — is a compile error rather than a silent zero.

Spawn bubbles anywhere along the bottom, and fling a survivor to a random spot on each hit.

```
Actions {
    spawn("bubble", random(40, 760), 40)
    setPosition("hero", random(0, 800), 60, 0)
    updateSpeed(random(3, 7))
}
```


## Random choices

randomAction { Actions { ... } Actions { ... } ... } is an action that picks ONE of its child Actions { ... } blocks uniformly at random each time it fires, and runs only that block. Where random(min, max) picks a number, randomAction picks a whole behaviour - a different spawn, color, or outcome. Each branch is a full Actions block, so a branch can hold several actions, not just one.

Each tap recolors the entity to one of three colors, chosen at random.

```
Actions {
    randomAction {
        Actions { setColor("ef4444") }
        Actions { setColor("22c55e") }
        Actions { setColor("3b82f6") }
    }
}
```


## Templates and spawning

A normal Entity is placed in the world when the game starts. An Entity with template(true) is the opposite — it never renders or ticks. Templates exist only as blueprints to be cloned at runtime by the spawn(...) action, which creates a fresh instance at a chosen position. Each clone gets a unique runtime id like "bubble:1", "bubble:2", ... so it can be addressed by setPosition or deleteEntity. The spawnHere(...) variant drops the clone at the position of the spawning entity instead, and can seed a clone var from a live bar.

A template entity. It is declared once, never appears on screen, and serves as the prototype for every spawned clone.

```
Entity {
    id("bubble")
    template(true)
    startPosition(0, 0)
    color("5bc0eb")
    Type { circle(28) }
    Movement { speed(2) AutoMove { direction(0, 1) } }
}
```

spawn("templateId", x, y) clones the named template at (x, y). It is most useful inside an OnElapsedTime body so a separate spawner entity fires clones on a fixed cadence.

```
Entity {
    id("spawner")
    startPosition(400, 40)
    color("00000000")    # fully transparent
    Type { circle(1) }
    Events {
        OnElapsedTime(1.5){
            repeatCount(-1)
            Actions { spawn("bubble", 400, 40) }
        }
    }
}
```

spawnHere("templateId", "seedVar"?, "seedBar"?) clones a template at the position of the source entity, not a fixed point -- so a moving entity leaves a trail. The optional second + third args copy a Bar's current value into a per-entity var on the new clone -- the second arg names the clone-side var to write, the third names the world-side Bar to read. Below: the clone's "life" var is seeded from the current value of the "length" Bar.

```
OnElapsedTime(0.34){
    repeatCount(-1)
    Actions {
        spawnHere("seg", "life", "length")
    }
}
```

spawnAtEmpty("templateId", cellSize, leftX, bottomY, cols, rows) picks ONE random empty cell of a (cols x rows) grid anchored at (leftX, bottomY) and spawns there. "Empty" only counts existing entities of the SAME template -- board tap-zones in a different template don't block the placement. Used by 2048's post-swipe tile drop.

```
Actions {
    spawnAtEmpty("tile", 130, 80, 200, 4, 4)
}
```


## Text entities

Type { text("string", size) } makes an entity that renders as text in its color, at the given font size. setText("id", "new") rewrites it at runtime — use it for prompts, instructions, or a game-over reason. text is left-anchored at the entity's startPosition (there is no centering primitive yet, so position it by hand). It collides and is tapped by its measured bounding box like any other shape.

A prompt that flips after an unpredictable delay and reacts to a tap.

```
Entity {
    id("prompt")
    startPosition(260, 560)
    color("e8e8e8")
    Type { text("Wait for it...", 40) }
    Events {
        OnTapped { Actions { setText("prompt", "Nice!") } }
        OnElapsedTime(random(1, 3)){
            repeatCount(-1)
            Actions { setText("prompt", "GO -- TAP!") }
        }
    }
}
```


## Tetromino pieces

Type { tetromino(cellSize, "letter") } gives an entity a multi-cell shape with up to four rotation states. Seven standard pieces baked in: "I", "O", "T", "L", "J", "S", "Z". The rotate(direction) action cycles through the rotations (1 = clockwise, -1 = counter-clockwise); an invalid new rotation (would overlap a wall or solid cell) reverts. GridMove on a tetromino does per-cell collision -- so an L-shape doesn't block on its empty corner.

A T-piece falling on a 40-px grid with all four direction signals wired up.

```
Entity {
    id("T")
    template(true)
    color("a855f7")
    Type { tetromino(40, "T") }
    Movement { GridMove(40, 0.6){ direction(0, -1) } }
    Events {
        OnSignal("rotateCw") { Actions { rotate(1) } }
        OnSignal("dropFast") { Actions { gridStep(0, -1) } }
    }
}
```

lock(templateId) splits a tetromino into one solid 1-cell entity per filled cell and deletes the source. The new cells inherit the piece's colour and get the given templateId so clearFullRows / clearFullColumns / clearFullBoxes can find them.

```
OnGridBlocked {
    Actions {
        lock("locked")
        clearFullRows(40, 10, 150, 200, 20, "locked", "rowCleared")
        broadcast("pieceLocked")
    }
}
```

Row / column / box clears: same shape, different scan axis. Signatures (comments below match the example):
  clearFullRows(cellSize, rowWidth, leftX, bottomY, rowCount, lockedTemplateId, clearSignal?)
  clearFullColumns(cellSize, colHeight, leftX, bottomY, colCount, lockedTemplateId, clearSignal?)
  clearFullBoxes(cellSize, boxSize, leftX, bottomY, cols, rows, lockedTemplateId, clearSignal?)

```
Actions {
    # 40 px cells, 10-wide rows starting at (150, 200), 20 rows.
    clearFullRows(40, 10, 150, 200, 20, "locked", "rowCleared")
    # Same grid, scanning 20-tall columns; 10 of them across.
    clearFullColumns(40, 10, 150, 200, 20, "locked", "colCleared")
    # 3x3 sub-boxes over a 9x9 board.
    clearFullBoxes(40, 3, 150, 200, 9, 9, "locked", "boxCleared")
}
```


## Joystick

joystick("targetId", radius, speed) is a top-level sugar that places a virtual analog stick (a hollow ring + a draggable thumb) at the bottom-left corner of the screen; in a world bigger than the window it stays there while the camera scrolls. The thumb tracks the finger, clamped to the ring radius; each frame, the joystick writes the target's velocity from the thumb offset, giving full analog control with magnitude AND diagonals. Pan-start within the ring grabs the joystick; pan-end snaps the thumb home (the target's velocity zeroes next frame). Only one joystick per level.

Drive a hero with the joystick. Give the target a Control { movementControl(0) } with speed(0): the joystick sets its velocity, speed(0) stops a finger from also dragging it, and a Control mover is blocked by solid(true) walls. An AutoMove target would move too, but AutoMove ignores solids, so it would pass straight through walls.

```
Window {
    size(8, 12)
    joystick("hero", 80, 6)
    Entity { id("hero") startPosition(400, 600)
        color("4ade80") Type { circle(30) }
        Movement { speed(0) Control { movementControl(0) } } }
    Entity { id("wall") startPosition(0, 800)
        color("1e293b") solid(true)
        Type { rectangle(800, 20) } }
}
```


## Bar (scoring)

Bar lives inside Window. It tracks an integer score and can trigger action blocks when the value hits specific targets -- perfect for win/lose conditions or progressive difficulty. Bar ids must be unique across the Window: two Bar("score"), or lives() + Bar("lives"), or two lives() all throw a compile error.

Start at 0. When the bar reaches 5, the player wins; when it reaches -3, the player loses.

```
Bar {
    initial(0)
    OnValue(5){ Actions { gameOver("YOU WIN") } }
    OnValue(-3){ Actions { gameOver("YOU LOSE") } }
}
```


## Named bars

A Window can declare more than one Bar, each with a leading name: Bar("lives" ...) Bar("score" ...). The classic unnamed Bar(...) still works - it is just the bar named "" (the default). updateBar("id", N) targets a named bar; the plain updateBar(N) form still means the default bar. Each OnValue trigger fires only on the bar it was declared inside, so "lives hits 0" and "score hits 50" stay independent.

Two counters. Losing all lives ends the game; score is tracked separately and shown stacked in the overlay.

```
Bar("lives"){
    initial(3)
    OnValue(0){ Actions { gameOver("DESTROYED") } }
}
Bar("score"){
    initial(0)
}
# elsewhere, inside some Events { ... } block:
# Actions { updateBar("lives", -1) }
# Actions { updateBar("score", 1) }
```


## whenBar guards

whenBar("id", "op", value){ Actions { ... } } runs its nested actions only when the named bar satisfies the comparison. op is one of "==", "!=", "<", "<=", ">", ">=" passed as a string; an unrecognised op never opens the guard. The guarded actions run with the same source entity as the whenBar itself, so a guarded deleteEntity() still self-targets. This turns the bar system into a small state machine - "if still alive at the time limit, win".

A survival timer that only awards the win if a life remains.

```
OnElapsedTime(30.0){
    Actions {
        whenBar("lives", ">", 0){
            Actions { gameOver("SURVIVED") } }
    }
}
```


## Per-entity variables

A Bar is one counter shared by the whole game. var("name", initial) declares a counter that belongs to a single entity -- and every spawned clone of a template gets its own independent copy. changeVar("name", delta) adds to that counter on the source entity. whenVar("name", "op", value){ Actions { ... } } is whenBar's per-entity sibling: it runs its block only when the source entity's counter satisfies the comparison. This is how you give many entities individual state -- per-enemy health, an ammo count, a behaviour flag.

An enemy with its own three-hit health. Each tap chips a point; at 0 it removes itself.

```
Entity {
    id("enemy")
    startPosition(300, 500)
    color("ef4444")
    Type { circle(30) }
    var("hp", 3)
    Events {
        OnTapped("enemy") {
            Actions {
                changeVar("hp", -1)
                whenVar("hp", "<=", 0){ Actions { deleteEntity() } }
            }
        }
    }
}
```


## Multi-level games

A Window with no Level { } children is a single-level game. A Window with one or more Level { } blocks is a multi-level game: each Level holds its own entities, and the engine plays them in declared order. The nextLevel() action advances; on the last level, nextLevel() wins the whole game. Bars declared on Window carry across levels (lives, scores accumulate); the entity list is rebuilt fresh each level. Used by Dungeon Escape, Grid Run, Brick Breaker and Minesweeper.

Two dungeons in sequence. The Window-level lives bar persists; clearing the first exit fires nextLevel() which rebuilds the entity list from Level 2.

```
Window {
    size(8, 12)
    lives(3, "-- CAUGHT --")
    Level {
        Entity { id("hero") startPosition(40, 40)
            color("4ade80") Type { rectangle(40, 40) }
            Movement { speed(7) Control { movementControl(0) } }
            Events { OnCollision("exit") {
                Actions { nextLevel() } } } }
        Entity { id("exit") startPosition(700, 1100)
            color("fbbf24") Type { rectangle(80, 80) } }
    }
    Level {
        Entity { id("hero") startPosition(40, 40)
            color("4ade80") Type { rectangle(40, 40) }
            Movement { speed(7) Control { movementControl(0) } }
            Events { OnCollision("exit") {
                Actions { gameOver("ESCAPED") } } } }
        Entity { id("exit") startPosition(700, 1100)
            color("fbbf24") Type { rectangle(80, 80) } }
    }
}
```


## Syntactic sugar

A handful of single-line keywords that expand at compile time into the same Entity/Event/Action graph you would write by hand. They add no new engine behaviour -- just shorter source and a smaller QR. Anywhere a sugar appears, the long form still works.

dpad("target") -- a four-pad steering rig plus the floor that keeps the player above it. Each pad sets the target entity's direction. Expands to five entities (floor + padUp/Down/Left/Right). Used by Snake, Pac-Man, Grid Run, Grid Dungeon.

```
Window {
    size(8, 12)
    dpad("hero")
    Entity { id("hero") startPosition(60, 440)
        color("22d3ee") Type { circle(40) }
        Movement { GridMove(100, 0.3){ direction(0, 0) } } }
}
```

dpad("target", "step") -- the discrete variant. Each pad fires a single gridStep on the target instead of setting a continuous direction. The right choice for Sokoban-style "one tap = one move".

```
dpad("hero", "step")
```

lives(N, "message") -- a bar named "lives" that starts at N and triggers gameOver(message) when it hits 0. Equivalent to a Bar("lives"){ initial(N) OnValue(0){ Actions { gameOver(...) } } }. Two lives() in one Window is a compile error. An optional third argument names a sound for the last life (see Sound effects).

```
Window {
    lives(3, "-- DESTROYED --", "lose")
    # ...rest of the game; decrement with
    # updateBar("lives", -1) on damage.
}
```

winAt(value, "message") (default bar) or winAt("barId", value, "message") (named bar) -- adds a single OnValue -> gameOver trigger to a bar. The message is optional (default "-- YOU WIN --"). The bar is auto-created if the Window does not declare it, and winAt can come before or after its Bar, so this is a one-liner for "win when score hits N". An optional last argument names a sound for the win.

```
Window {
    winAt(10, "HIGH SCORE", "win")
    # ...later, updateBar(1) per tap
}
```

hostile("target", "message") -- on an entity, adds OnCollision(target) -> gameOver. Drops the kill-on-touch from the hero's event list onto each enemy itself. An optional third argument names a sound for the catch.

```
Entity { id("guard") startPosition(400, 400)
    color("ef4444") Type { circle(26) }
    hostile("hero", "-- CAUGHT --", "lose")
    Movement { speed(3) chase("hero") } }
```

HostilePatrol("id", "target", "killMsg", speed, "color") + waypoint(...) children + optional dwell(N) -- the hostile + Path + catch-FX combo in one block, with sound("lose") baked into the catch. Replaces ~10 lines per patrol guard.

```
HostilePatrol("guard1", "hero", "-- CAUGHT --", 3, "ef4444"){
    waypoint(180, 800)
    waypoint(700, 800)
    dwell(1)
}
```

key("signal", "picker") + door("signal") -- the key broadcasts the signal and deletes itself on collision with the picker (default "hero"). The door listens for the signal, deletes itself, and is auto-solid. picker is optional. Each takes an optional sound last: the key's plays on pickup, the door's as it opens.

```
Entity { id("key") startPosition(372, 540)
    color("facc15") Type { rectangle(48, 48) }
    key("unlocked", "hero", "pickup") }
Entity { id("door") startPosition(440, 940)
    color("7c3aed") Type { rectangle(360, 34) }
    door("unlocked", "clang") }
```

paddle("id", x, y, w, h) -- a horizontal-only Control rectangle positioned at (x, y) with size (w, h). Defaults: cyan, speed 8. A brick-breaker paddle in one line.

```
paddle("ship", 420, 80, 80, 24)
```

goal("id", x, y, w, h, "winMsg") -- a gold rectangle that ends the run on hero contact, with baked sound("win") + flash + burst + shake polish. Listens for the entity named "hero".

```
goal("flag", 700, 1100, 80, 60, "-- REACHED IT --")
```

joystick("targetId", radius, speed) -- virtual analog stick at the bottom-left corner. Drives the target's velocity proportional to thumb displacement (diagonals + magnitude included). See "Joystick" section.

```
joystick("hero", 80, 6)
```

wallBox(x, y, w, h, "color", thickness) -- four solid walls (wallL/R/T/B) framing the rect. Defaults: color "1e293b", thickness 20. One per level.

```
wallBox(0, 0, 700, 1200, "1e293b", 10)
```


## Common patterns

Composable building blocks the demos lean on heavily. Each example below is a complete, runnable program — copy any of them into the editor and press Play.

**Wall-bouncing ball.** A single circle that bounces off every wall forever. The simplest possible program.

```
Window {
    size(8, 8)
    background("0a0e27")
    Entity {
        id("ball")
        startPosition(400, 400)
        color("4ade80")
        Type { circle(20) }
        Movement { speed(4) AutoMove { direction(1, 1) } }
        Events {
            OnCollision { Actions { bounceBack() } }
        }
    }
}
```

**Tap-to-score button.** A coloured square. Each tap adds one to the score. Reach 10 to win.

```
Window {
    size(8, 8)
    background("121826")
    Bar {
        initial(0)
        OnValue(10){ Actions { gameOver("HIGH SCORE") } }
    }
    Entity {
        id("btn")
        startPosition(350, 350)
        color("f59e0b")
        Type { rectangle(100, 100) }
        Events {
            OnTapped("btn") { Actions { updateBar(1) } }
        }
    }
}
```

**Falling obstacle that respawns at the top.** The "infinite spawner" trick: when a falling AutoMove entity hits the floor, setPosition teleports it back up. Lets a finite entity list act like an endless stream.

```
Window {
    size(8, 12)
    background("0a0e27")
    Entity {
        id("rock")
        startPosition(400, 1100)
        color("ef4444")
        Type { triangle(40, "down") }
        Movement { speed(3) AutoMove { direction(0, -1) } }
        Events {
            OnCollision { Actions { setPosition("rock", 400, 1140, 0) } }
        }
    }
}
```

**Paddle + bouncing ball.** The brick-breaker skeleton: a draggable paddle at the bottom plus a ball that bounces off it AND the walls. The paddle uses movementControl(2) for horizontal-only drag.

```
Window {
    size(8, 10)
    background("121826")
    Entity {
        id("paddle")
        startPosition(360, 60)
        color("38bdf8")
        Type { rectangle(120, 20) }
        Movement { speed(5) Control { movementControl(2) } }
    }
    Entity {
        id("ball")
        startPosition(400, 500)
        color("fbbf24")
        Type { circle(15) }
        Movement { speed(4) AutoMove { direction(1, -1) } }
        Events {
            OnCollision("paddle") { Actions { bounceBack() } }
            OnCollision { Actions { bounceBack() } }
        }
    }
}
```


## Engine notes (from the engine's source, not in the Docs)

- A bar's OnValue actions (also when a ramp passes the value) run with no entity: actions that act on the running entity (setColor, changeVar, spawnHere, burst or flash without an id, ...) do nothing there. Name the entity, or broadcast a signal.
- Wave actions run on a hidden entity at (-1, -1): spawnHere there spawns off screen; spawn with coordinates.
- OnValue fires only when an update lands exactly on that value: updateBar("x", -3) can jump over OnValue(0). loopCount(3) then updateBar("x", -1) steps through every value.
- A collision fires once when the overlap starts, not every frame.
- restart() puts every bar back to its value at the start of the level, lives included.
- A disabled entity (setEnable(false)) still runs OnSignal and its pending delays, but not its timers, taps or collisions. A hidden one (setVisible(false)) can still be tapped.
- setColor only colours the entity that runs it; to recolour another, broadcast a signal it listens to.
- OnTapped("template") never matches clones (their ids are template:1, template:2, ...): give the template a plain OnTapped.

## Placement table: which keywords go inside which block

(collected from the built-in games, the examples above and the compiler)

In a game with Level blocks, every Entity (and dpad, joystick, paddle, goal, HostilePatrol, wallBox, Wave) must sit inside a Level: the ones written directly in the Window are ignored. Size, colours, music, gravity, Bars, lives and winAt stay in the Window.

- Actions: applyImpulse, bounceBack, broadcast, burst, changeVar, clearFullBoxes, clearFullColumns, clearFullRows, delay, deleteCollider, deleteEntity, flash, gameOver, gridStep, lock, loopCount, nextLevel, ramp, randomAction, restart, retarget, rotate, setColor, setDirection, setEnable, setOpacity, setPosition, setScale, setText, setVelocity, setVisible, shake, slide, slideAll, sound, spawn, spawnAtEmpty, spawnHere, updateBar, updateSpeed, whenBar, whenVar
- Entity: Events, Movement, Type, color, door, facing, gravity, hostile, id, key, oneWay, pushable, solid, startPosition, template, var
- Events: OnCollision, OnElapsedTime, OnGridBlocked, OnGridStep, OnSignal, OnSwipe, OnTapped, Sequence
- Level: Entity, HostilePatrol, Wave, dpad, goal, joystick, paddle, wallBox
- Movement: AutoMove, Control, GridMove, Path, chase, speed
- Type: circle, cross, diamond, polygon, rectangle, ring, star, tetromino, text, triangle
- Window: Bar, Entity, HostilePatrol, Level, Wave, background, cameraFollow, dpad, goal, gravity, joystick, lives, music, paddle, size, wallBox, winAt, worldSize
- AutoMove: direction
- Bar: OnValue, initial
- Control: movementControl
- delay: Actions
- GridMove: direction
- HostilePatrol: dwell, waypoint
- OnCollision: Actions
- OnElapsedTime: Actions, repeatCount
- OnGridBlocked: Actions
- OnGridStep: Actions
- OnSignal: Actions
- OnSwipe: Actions
- OnTapped: Actions
- OnValue: Actions
- Path: dwell, waypoint
- Phase: Actions
- randomAction: Actions
- Sequence: Phase
- Wave: Actions
- whenBar: Actions
- whenVar: Actions
