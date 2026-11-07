# The game language: reference card

A program is one Window block. Grammar: `keyword(parameters) { child blocks }`, both parts optional.
Parentheses hold only strings, numbers, true/false and random(min, max); everything nested goes in braces.
Keywords are case-insensitive; a keyword in the wrong block is silently ignored (see the placement table).
# starts a comment. Positions and sizes are pixels from the bottom-left corner, y up; startPosition is the
bottom-left corner. size(w, h) and worldSize are in units of 100 px. speed() is in units of 100 px/s,
setVelocity / applyImpulse in px/s; times in seconds. Colours: "RRGGBB", "AARRGGBB" or short "RGB".
Many actions act on the entity whose event fired unless an id comes first: setVelocity, applyImpulse,
gridStep, setScale, setOpacity, setVisible, setEnable, deleteEntity, flash, burst.
A template(true) Entity is a blueprint: spawn("id", x, y) / spawnHere("id") make clones; a template's id
matches all its clones in OnCollision. broadcast("name") runs every OnSignal("name").

## One example call per keyword

- Actions()
- applyImpulse("ball", 220, 480)
- AutoMove { ... }
- background("87CEEB")
- Bar("m")
- bounceBack()
- broadcast("r1")
- burst(8, "fbbf24")
- cameraFollow("hero")
- changeVar("hp", 1)
- chase("gh")
- circle(1)
- clearFullBoxes(40, 3, 150, 200, 9, 9, "locked", "boxCleared")
- clearFullColumns(40, 10, 150, 200, 20, "locked", "colCleared")
- clearFullRows(40, 10, 150, 200, 20, "locked", "rowCleared")
- color("fff")
- Control { ... }
- cross(40, 40, 12)
- delay(1)
- deleteCollider()
- deleteEntity()
- diamond(60, 60)
- direction(0, 0)
- door("unlock1")
- dpad("pac")
- dwell(1)
- Entity { ... }
- Events { ... }
- facing("left")
- flash("ef4444", 1)
- gameOver("WALL")
- goal("flag", 700, 1100, 80, 60, "-- REACHED IT --")
- gravity(0)
- GridMove(40, 0.6)
- gridStep(1, 0)
- hostile("hero", "-- CAUGHT --", "lose")
- HostilePatrol("guard1", "hero", "-- CAUGHT --", 3, "ef4444")
- id("T")
- initial(0)
- joystick("hero", 80, 6)
- key("unlock1", "hero")
- Level { ... }
- lives(3, "-- CAUGHT --", "lose")
- lock("locked")
- loopCount(3)
- Movement { ... }
- movementControl(2)
- music("bg_3")
- nextLevel()
- OnCollision("gm")
- OnElapsedTime(0)
- oneWay(true)
- OnGridBlocked { ... }
- OnGridStep { ... }
- OnSignal("gP")
- OnSwipe("up")
- OnTapped("btn")
- OnValue(5)
- paddle("ship", 420, 80, 80, 24)
- Path { ... }
- Phase(1, 2)
- polygon(6, 30)
- pushable(true)
- ramp("difficulty", 1, 10, 30)
- randomAction { ... }
- rectangle(0, 0)
- repeatCount(3)
- restart()
- retarget("bt")
- ring(15, 8)
- rotate(1)
- Sequence { ... }
- setColor("4ade80")
- setDirection("hero", 1, 0)
- setEnable(true)
- setOpacity(0.3)
- setPosition("ball", 100, 500, 0)
- setScale("ball", 1.5)
- setText("c00", "1")
- setVelocity("inv", 170, 0)
- setVisible(false)
- shake(6, 0.3)
- size(8, 8)
- slide(-1, 0, "value", "recolor")
- slideAll(1, 0, 130, "tile", "value", "recolor")
- solid(true)
- sound("win")
- spawn("vl", 0, 0)
- spawnAtEmpty("tile", 130, 80, 200, 4, 4)
- spawnHere("p1")
- speed(6)
- star(5, 4, 2)
- startPosition(0, 0)
- template(true)
- tetromino(40, "T")
- text("🐎", 58)
- triangle(50, "up")
- Type { ... }
- updateBar("m", 7)
- updateSpeed(7)
- var("z", 0)
- wallBox(0, 0, 700, 1200, "1e293b", 10)
- Wave(6, 1.3)
- waypoint(90, 900)
- whenBar("boss", ">", 0)
- whenVar("hp", "<", 0)
- winAt(10, "HIGH SCORE", "win")
- Window()
- worldSize(40, 9)

Sound names: beep, bell, blip, bubble, bump, buzz, clang, click, coin, death, glass, hurt, jump, laser, lose, pickup, pop, sparkle, thud, win, win2
Music tracks: bg_1, bg_2, bg_3, bg_4, bg_5, bg_6, bg_ambient, bg_nature

## A complete program

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

