# Sort Line: rubbish rides the belt into a diverter. Swipe up to send it to the paper chute,
# down to the glass chute, right to let it run on to the can bin. A wrong bin or a spill costs a life.
# The morning shift has no glass; the night shift has everything, faster.
Window {
    size(10, 10)
    background("1f2937")
    music("bg_6")
    gravity(0)

    lives(3, "-- THE BATCH IS CONTAMINATED --", "lose")
    Bar("sorted") { initial(0)
        OnValue(12) { Actions { sound("win2") shake(4, 0.4) nextLevel() } }
        OnValue(32) { Actions { sound("win") shake(6, 0.5) gameOver("-- SHIFT DONE, EVERYTHING SORTED --") } } }
    Bar("gate") { initial(0)
        OnValue(0) { Actions { broadcast("toCans") } }
        OnValue(1) { Actions { broadcast("toPaper") } }
        OnValue(2) { Actions { broadcast("toGlass") } } }

    Level {
        Wave(30, 2.2) { Actions { randomAction {
            Actions { spawn("paper", 10, 485) } Actions { spawn("can", 10, 485) } } } }
        Wave(90, 1.6) { Actions { randomAction {
            Actions { spawn("paper", 10, 485) } Actions { spawn("can", 10, 485) } } } }

        Entity { id("label") startPosition(30, 940) color("e5e7eb") Type { text("MORNING SHIFT: paper and cans", 32) }
            Events {
                OnSignal("toCans") { Actions { setText("label", "Gate: straight on to the cans") } }
                OnSignal("toPaper") { Actions { setText("label", "Gate: up to the paper chute") } }
                OnSignal("toGlass") { Actions { setText("label", "Gate: down - no glass bin today!") } } } }
        Entity { id("belt") startPosition(0, 470) color("4b5563") Type { rectangle(1000, 60) }
            Events {
                OnSwipe("up") { Actions { whenBar("gate", "==", 0) { Actions { updateBar("gate", 1) } } whenBar("gate", "==", 2) { Actions { updateBar("gate", -1) } } sound("click") } }
                OnSwipe("down") { Actions { whenBar("gate", "==", 0) { Actions { updateBar("gate", 2) } } whenBar("gate", "==", 1) { Actions { updateBar("gate", 1) } } sound("click") } }
                OnSwipe("right") { Actions { whenBar("gate", "==", 1) { Actions { updateBar("gate", -1) } } whenBar("gate", "==", 2) { Actions { updateBar("gate", -2) } } sound("click") } } } }
        Entity { id("diverter") startPosition(560, 440) color("fbbf24") Type { rectangle(20, 120) }
            Events {
                OnSignal("toCans") { Actions { setColor("fbbf24") } }
                OnSignal("toPaper") { Actions { setColor("f8fafc") } }
                OnSignal("toGlass") { Actions { setColor("22d3ee") } } } }
        Entity { id("paperBin") startPosition(460, 880) color("1d4ed8") Type { rectangle(240, 80) } }
        Entity { id("canBin") startPosition(900, 430) color("b45309") Type { rectangle(100, 140) } }

        Entity { id("paper") template(true) color("f8fafc") Type { rectangle(40, 30) }
            Movement { speed(1.8) AutoMove { direction(1, 0) } }
            Events {
                OnCollision("diverter") { Actions {
                    whenBar("gate", "==", 1) { Actions { setVelocity(0, 220) } }
                    whenBar("gate", "==", 2) { Actions { setVelocity(0, -220) } } } }
                OnCollision("paperBin") { Actions { sound("coin") burst(8, "f8fafc") updateBar("sorted", 1) deleteEntity() } }
                OnCollision("canBin") { Actions { sound("hurt") shake(5, 0.3) flash("canBin", "ef4444", 0.3) updateBar("lives", -1) deleteEntity() } }
                OnCollision { Actions { sound("thud") burst(8, "94a3b8") updateBar("lives", -1) deleteEntity() } } } }
        Entity { id("can") template(true) color("94a3b8") Type { circle(20) }
            Movement { speed(1.8) AutoMove { direction(1, 0) } }
            Events {
                OnCollision("diverter") { Actions {
                    whenBar("gate", "==", 1) { Actions { setVelocity(0, 220) } }
                    whenBar("gate", "==", 2) { Actions { setVelocity(0, -220) } } } }
                OnCollision("canBin") { Actions { sound("clang") burst(8, "cbd5e1") updateBar("sorted", 1) deleteEntity() } }
                OnCollision("paperBin") { Actions { sound("hurt") shake(5, 0.3) flash("paperBin", "ef4444", 0.3) updateBar("lives", -1) deleteEntity() } }
                OnCollision { Actions { sound("thud") burst(8, "94a3b8") updateBar("lives", -1) deleteEntity() } } } }
    }

    Level {
        Wave(20, 1.6) { Actions { randomAction {
            Actions { spawn("paper", 10, 485) } Actions { spawn("can", 10, 485) } Actions { spawn("jar", 10, 485) } } } }
        Wave(40, 1.1) { Actions { randomAction {
            Actions { spawn("paper", 10, 485) } Actions { spawn("can", 10, 485) } Actions { spawn("jar", 10, 485) } } } }
        Wave(90, 0.85) { Actions { randomAction {
            Actions { spawn("paper", 10, 485) } Actions { spawn("can", 10, 485) } Actions { spawn("jar", 10, 485) } } } }

        Entity { id("label") startPosition(30, 940) color("e5e7eb") Type { text("NIGHT SHIFT: paper, cans and glass", 32) }
            Events {
                OnSignal("toCans") { Actions { setText("label", "Gate: straight on to the cans") } }
                OnSignal("toPaper") { Actions { setText("label", "Gate: up to the paper chute") } }
                OnSignal("toGlass") { Actions { setText("label", "Gate: down to the glass chute") } } } }
        Entity { id("belt") startPosition(0, 470) color("374151") Type { rectangle(1000, 60) }
            Events {
                OnSwipe("up") { Actions { whenBar("gate", "==", 0) { Actions { updateBar("gate", 1) } } whenBar("gate", "==", 2) { Actions { updateBar("gate", -1) } } sound("click") } }
                OnSwipe("down") { Actions { whenBar("gate", "==", 0) { Actions { updateBar("gate", 2) } } whenBar("gate", "==", 1) { Actions { updateBar("gate", 1) } } sound("click") } }
                OnSwipe("right") { Actions { whenBar("gate", "==", 1) { Actions { updateBar("gate", -1) } } whenBar("gate", "==", 2) { Actions { updateBar("gate", -2) } } sound("click") } } } }
        Entity { id("diverter") startPosition(560, 440) color("fbbf24") Type { rectangle(20, 120) }
            Events {
                OnElapsedTime { Actions { whenBar("gate", "==", 1) { Actions { setColor("f8fafc") } } whenBar("gate", "==", 2) { Actions { setColor("22d3ee") } } } }
                OnSignal("toCans") { Actions { setColor("fbbf24") } }
                OnSignal("toPaper") { Actions { setColor("f8fafc") } }
                OnSignal("toGlass") { Actions { setColor("22d3ee") } } } }
        Entity { id("paperBin") startPosition(460, 880) color("1d4ed8") Type { rectangle(240, 80) } }
        Entity { id("canBin") startPosition(900, 430) color("b45309") Type { rectangle(100, 140) } }
        Entity { id("jarBin") startPosition(460, 40) color("15803d") Type { rectangle(240, 80) } }

        Entity { id("paper") template(true) color("f8fafc") Type { rectangle(40, 30) }
            Movement { speed(2.3) AutoMove { direction(1, 0) } }
            Events {
                OnCollision("diverter") { Actions {
                    whenBar("gate", "==", 1) { Actions { setVelocity(0, 260) } }
                    whenBar("gate", "==", 2) { Actions { setVelocity(0, -260) } } } }
                OnCollision("paperBin") { Actions { sound("coin") burst(8, "f8fafc") updateBar("sorted", 1) deleteEntity() } }
                OnCollision("canBin") { Actions { sound("hurt") shake(5, 0.3) flash("canBin", "ef4444", 0.3) updateBar("lives", -1) deleteEntity() } }
                OnCollision("jarBin") { Actions { sound("hurt") shake(5, 0.3) flash("jarBin", "ef4444", 0.3) updateBar("lives", -1) deleteEntity() } } } }
        Entity { id("can") template(true) color("94a3b8") Type { circle(20) }
            Movement { speed(2.3) AutoMove { direction(1, 0) } }
            Events {
                OnCollision("diverter") { Actions {
                    whenBar("gate", "==", 1) { Actions { setVelocity(0, 260) } }
                    whenBar("gate", "==", 2) { Actions { setVelocity(0, -260) } } } }
                OnCollision("canBin") { Actions { sound("clang") burst(8, "cbd5e1") updateBar("sorted", 1) deleteEntity() } }
                OnCollision("paperBin") { Actions { sound("hurt") shake(5, 0.3) flash("paperBin", "ef4444", 0.3) updateBar("lives", -1) deleteEntity() } }
                OnCollision("jarBin") { Actions { sound("hurt") shake(5, 0.3) flash("jarBin", "ef4444", 0.3) updateBar("lives", -1) deleteEntity() } } } }
        Entity { id("jar") template(true) color("22d3ee") Type { diamond(36, 44) }
            Movement { speed(2.3) AutoMove { direction(1, 0) } }
            Events {
                OnCollision("diverter") { Actions {
                    whenBar("gate", "==", 1) { Actions { setVelocity(0, 260) } }
                    whenBar("gate", "==", 2) { Actions { setVelocity(0, -260) } } } }
                OnCollision("jarBin") { Actions { sound("glass") burst(10, "a5f3fc") updateBar("sorted", 1) deleteEntity() } }
                OnCollision("paperBin") { Actions { sound("hurt") shake(5, 0.3) flash("paperBin", "ef4444", 0.3) updateBar("lives", -1) deleteEntity() } }
                OnCollision("canBin") { Actions { sound("hurt") shake(5, 0.3) flash("canBin", "ef4444", 0.3) updateBar("lives", -1) deleteEntity() } } } }
    }
}
