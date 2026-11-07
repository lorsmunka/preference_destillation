Window {
    size(8, 10) worldSize(8, 20) cameraFollow("hero")
    background("1e1b4b") music("bg_1") gravity(0)

    lives(3, "-- SWEPT OFF THE STAIRS --", "lose")
    Bar("oil") { initial(0)
        OnValue(3) { Actions { broadcast("lampReady") } } }

    # Every can makes the keeper slower on the stairs.
    Entity { id("hero") startPosition(380, 80) color("fbbf24") Type { rectangle(40, 52) }
        Movement { speed(5) Control { movementControl(0) } }
        Events {
            OnCollision("can") { Actions { sound("pickup") updateBar("oil", 1) burst(8, "a16207") deleteCollider()
                whenBar("oil", "==", 1) { Actions { updateSpeed(4) } }
                whenBar("oil", "==", 2) { Actions { updateSpeed(3.2) } }
                whenBar("oil", "==", 3) { Actions { updateSpeed(2.6) } } } }
            OnSignal("lampReady") { Actions { sound("bell") flash("fde047", 0.5) } } } }

    Entity { id("f1") startPosition(0, 400) color("475569") solid(true) Type { rectangle(560, 30) } }
    Entity { id("f2") startPosition(240, 800) color("475569") solid(true) Type { rectangle(560, 30) } }
    Entity { id("f3") startPosition(0, 1200) color("475569") solid(true) Type { rectangle(560, 30) } }
    Entity { id("f4") startPosition(240, 1600) color("475569") solid(true) Type { rectangle(560, 30) } }
    Entity { id("topL") startPosition(0, 1800) color("475569") solid(true) Type { rectangle(300, 30) } }
    Entity { id("topR") startPosition(500, 1800) color("475569") solid(true) Type { rectangle(300, 30) } }
    Entity { id("hatch") startPosition(300, 1800) color("92400e") solid(true) Type { rectangle(200, 30) }
        Events { OnSignal("lampReady") { Actions { sound("clang") burst(14, "fbbf24") setEnable(false) } } } }
    Entity { id("hatchSign") startPosition(240, 1700) color("c7d2fe") Type { text("HATCH: bring 3 cans of oil", 26) }
        Events { OnSignal("lampReady") { Actions { setText("hatchSign", "HATCH OPEN - light the lamp") setColor("fde047") } } } }
    goal("lamp", 330, 1880, 140, 100, "-- THE LIGHTHOUSE SHINES AGAIN --")

    Entity { id("can") template(true) color("a16207") Type { rectangle(30, 40) } }
    Entity { id("cellar") startPosition(0, 0) color("00000000") Type { circle(1) }
        Events { OnElapsedTime { Actions { spawn("can", 680, 200) spawn("can", 60, 620) spawn("can", 700, 1420) } } } }

    # Sea spray bursts in through the windows and knocks you about.
    Entity { id("spray") template(true) color("bae6fd") gravity(0) Type { rectangle(70, 26) }
        Movement { speed(4.5) AutoMove { direction(1, 0) } }
        Events {
            OnCollision("hero") { Actions { sound("hurt") shake(6, 0.3) flash("hero", "38bdf8", 0.3) updateBar("lives", -1) deleteEntity() } }
            OnCollision { Actions { deleteEntity() } } } }
    Entity { id("storm") startPosition(0, 0) color("00000000") Type { circle(1) }
        Events { OnElapsedTime(random(0.8, 1.6)) { repeatCount(-1) Actions {
            randomAction {
                Actions { spawn("spray", 0, random(460, 760)) }
                Actions { spawn("spray", 0, random(860, 1160)) }
                Actions { spawn("spray", 0, random(1260, 1560)) } } } } } }
}
