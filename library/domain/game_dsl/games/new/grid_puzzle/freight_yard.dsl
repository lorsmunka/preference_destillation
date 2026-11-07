# Freight yard, seen from above. Wagons roll in from the right and run left until they buffer up
# against the stop or the wagons already parked. Tap UP and DOWN to switch the rolling wagon to
# another track, TURN to swing it round, NUDGE to roll it one cell on. A column filled across
# all six tracks is a train: it leaves at once. Wagons left standing in the entry throat block it.
Window {
    size(8, 10)
    background("292524")
    music("bg_3")
    gravity(0)

    Bar("trains") { initial(0) }
    winAt("trains", 5, "-- FIVE TRAINS DISPATCHED --", "win")

    Entity { id("bufferStop") startPosition(0, 280) color("b91c1c") Type { rectangle(40, 440) } solid(true) }
    Entity { id("fenceS") startPosition(0, 280) color("57534e") Type { rectangle(800, 40) } solid(true) }
    Entity { id("fenceN") startPosition(0, 680) color("57534e") Type { rectangle(800, 40) } solid(true) }
    Entity { id("throat") startPosition(520, 320) color("22fde047") Type { rectangle(240, 360) }
        Events { OnCollision("wagon") { Actions { sound("lose") shake(8, 0.6) gameOver("-- THE YARD IS BLOCKED --") } } } }

    Entity { id("wagon") template(true) startPosition(0, 0) color("a8a29e") Type { rectangle(60, 60) } solid(true) }

    Entity { id("wagonI") template(true) startPosition(0, 0) color("38bdf8") Type { tetromino(60, "I") }
        Movement { GridMove(60, 0.5) { direction(-1, 0) } }
        Events {
            OnSignal("trackUp") { Actions { gridStep(0, 1) } }
            OnSignal("trackDown") { Actions { gridStep(0, -1) } }
            OnSignal("couple") { Actions { rotate(-1) } }
            OnSignal("nudge") { Actions { gridStep(-1, 0) } }
            OnGridBlocked { Actions { sound("clang") burst(8, "38bdf8") lock("wagon")
                clearFullColumns(60, 6, 40, 320, 8, "wagon", "trainOut") broadcast("nextWagon") } } } }
    Entity { id("wagonO") template(true) startPosition(0, 0) color("facc15") Type { tetromino(60, "O") }
        Movement { GridMove(60, 0.5) { direction(-1, 0) } }
        Events {
            OnSignal("trackUp") { Actions { gridStep(0, 1) } }
            OnSignal("trackDown") { Actions { gridStep(0, -1) } }
            OnSignal("couple") { Actions { rotate(-1) } }
            OnSignal("nudge") { Actions { gridStep(-1, 0) } }
            OnGridBlocked { Actions { sound("clang") burst(8, "facc15") lock("wagon")
                clearFullColumns(60, 6, 40, 320, 8, "wagon", "trainOut") broadcast("nextWagon") } } } }
    Entity { id("wagonL") template(true) startPosition(0, 0) color("fb923c") Type { tetromino(60, "L") }
        Movement { GridMove(60, 0.5) { direction(-1, 0) } }
        Events {
            OnSignal("trackUp") { Actions { gridStep(0, 1) } }
            OnSignal("trackDown") { Actions { gridStep(0, -1) } }
            OnSignal("couple") { Actions { rotate(-1) } }
            OnSignal("nudge") { Actions { gridStep(-1, 0) } }
            OnGridBlocked { Actions { sound("clang") burst(8, "fb923c") lock("wagon")
                clearFullColumns(60, 6, 40, 320, 8, "wagon", "trainOut") broadcast("nextWagon") } } } }
    Entity { id("wagonT") template(true) startPosition(0, 0) color("c084fc") Type { tetromino(60, "T") }
        Movement { GridMove(60, 0.5) { direction(-1, 0) } }
        Events {
            OnSignal("trackUp") { Actions { gridStep(0, 1) } }
            OnSignal("trackDown") { Actions { gridStep(0, -1) } }
            OnSignal("couple") { Actions { rotate(-1) } }
            OnSignal("nudge") { Actions { gridStep(-1, 0) } }
            OnGridBlocked { Actions { sound("clang") burst(8, "c084fc") lock("wagon")
                clearFullColumns(60, 6, 40, 320, 8, "wagon", "trainOut") broadcast("nextWagon") } } } }
    Entity { id("wagonS") template(true) startPosition(0, 0) color("4ade80") Type { tetromino(60, "S") }
        Movement { GridMove(60, 0.5) { direction(-1, 0) } }
        Events {
            OnSignal("trackUp") { Actions { gridStep(0, 1) } }
            OnSignal("trackDown") { Actions { gridStep(0, -1) } }
            OnSignal("couple") { Actions { rotate(-1) } }
            OnSignal("nudge") { Actions { gridStep(-1, 0) } }
            OnGridBlocked { Actions { sound("clang") burst(8, "4ade80") lock("wagon")
                clearFullColumns(60, 6, 40, 320, 8, "wagon", "trainOut") broadcast("nextWagon") } } } }

    Entity { id("signalBox") startPosition(40, 760) color("e7e5e4") Type { text("FILL A COLUMN OF SIX", 36) }
        Events {
            OnElapsedTime(0.5) { Actions { broadcast("nextWagon") } }
            OnSignal("nextWagon") { Actions { randomAction {
                Actions { spawn("wagonI", 520, 440) }
                Actions { spawn("wagonO", 520, 440) }
                Actions { spawn("wagonL", 520, 440) }
                Actions { spawn("wagonT", 520, 440) }
                Actions { spawn("wagonS", 520, 440) } } } }
            OnSignal("trainOut") { Actions { sound("bell") shake(4, 0.4) flash("4ade80", 0.5) updateBar("trains", 1) } } } }

    Entity { id("leverUp") startPosition(40, 120) color("2563eb") Type { rectangle(170, 120) }
        Events { OnTapped("leverUp") { Actions { sound("click") flash("ffffff", 0.08) broadcast("trackUp") } } } }
    Entity { id("leverDown") startPosition(225, 120) color("2563eb") Type { rectangle(170, 120) }
        Events { OnTapped("leverDown") { Actions { sound("click") flash("ffffff", 0.08) broadcast("trackDown") } } } }
    Entity { id("leverTurn") startPosition(410, 120) color("7c3aed") Type { rectangle(170, 120) }
        Events { OnTapped("leverTurn") { Actions { sound("blip") flash("ffffff", 0.08) broadcast("couple") } } } }
    Entity { id("leverNudge") startPosition(595, 120) color("059669") Type { rectangle(170, 120) }
        Events { OnTapped("leverNudge") { Actions { sound("click") flash("ffffff", 0.08) broadcast("nudge") } } } }
    Entity { id("labelUp") startPosition(95, 160) color("e0e7ff") Type { text("UP", 36) } }
    Entity { id("labelDown") startPosition(255, 160) color("e0e7ff") Type { text("DOWN", 36) } }
    Entity { id("labelTurn") startPosition(445, 160) color("ede9fe") Type { text("TURN", 36) } }
    Entity { id("labelNudge") startPosition(615, 160) color("d1fae5") Type { text("NUDGE", 34) } }
}
