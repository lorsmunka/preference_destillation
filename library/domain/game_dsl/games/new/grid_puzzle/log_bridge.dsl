# A beaver pushes logs into the river. A log that lands in open water sinks into a plank;
# a log pushed onto a plank just rests on it. Water is spawned before the logs and the
# beaver so they are drawn on top of it.
Window {
    size(8, 12)
    background("4d7c0f")
    music("bg_nature")
    gravity(0)

    Level {
        Entity { id("titleL1") startPosition(20, 1150) color("ecfccb") Type { text("CREEK 1/2", 30) } }
        Entity { id("retry1") startPosition(640, 60) color("fef3c7") Type { text("RETRY", 36) }
            Events { OnTapped("retry1") { Actions { sound("click") restart() } } } }
        dpad("hero", "step")

        Entity { id("water") template(true) startPosition(0, 0) color("0284c7") Type { rectangle(100, 100) } var("bridged", 0)
            Events {
                OnCollision("log") { Actions { whenVar("bridged", "==", 0) { Actions {
                    changeVar("bridged", 1) setColor("92400e") sound("bubble") burst(10, "7dd3fc") deleteCollider() } } } }
                OnCollision("hero") { Actions { whenVar("bridged", "==", 0) { Actions {
                    sound("lose") shake(6, 0.4) burst(16, "7dd3fc") gameOver("-- SWEPT AWAY --") } } } } } }
        Entity { id("log") template(true) startPosition(0, 0) color("a16207") Type { rectangle(90, 70) } pushable(true)
            Events { OnGridStep { Actions { sound("thud") } } } }
        Entity { id("hero") template(true) startPosition(0, 0) color("78350f") Type { circle(40) }
            Movement { GridMove(100, 0.2) { direction(0, 0) } }
            Events { OnCollision("lodge") { Actions { sound("win2") flash("fde68a", 0.3) burst(16, "fde68a") nextLevel() } } } }

        Entity { id("lodge") startPosition(400, 1100) color("713f12") Type { polygon(6, 50) } }

        Entity { id("riverbed") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { OnElapsedTime { Actions {
                spawn("water", 0, 700) spawn("water", 100, 700) spawn("water", 200, 700) spawn("water", 300, 700)
                spawn("water", 400, 700) spawn("water", 500, 700) spawn("water", 600, 700) spawn("water", 700, 700)
                spawn("water", 0, 800) spawn("water", 100, 800) spawn("water", 200, 800) spawn("water", 300, 800)
                spawn("water", 400, 800) spawn("water", 500, 800) spawn("water", 600, 800) spawn("water", 700, 800)
                spawn("log", 205, 515) spawn("log", 505, 515)
                spawn("hero", 310, 410) } } } }
    }

    Level {
        Entity { id("titleL2") startPosition(20, 1150) color("ecfccb") Type { text("CREEK 2/2", 30) } }
        Entity { id("retry2") startPosition(640, 60) color("fef3c7") Type { text("RETRY", 36) }
            Events { OnTapped("retry2") { Actions { sound("click") restart() } } } }
        dpad("hero", "step")

        Entity { id("water") template(true) startPosition(0, 0) color("0369a1") Type { rectangle(100, 100) } var("bridged", 0)
            Events {
                OnCollision("log") { Actions { whenVar("bridged", "==", 0) { Actions {
                    changeVar("bridged", 1) setColor("92400e") sound("bubble") burst(10, "7dd3fc") deleteCollider() } } } }
                OnCollision("hero") { Actions { whenVar("bridged", "==", 0) { Actions {
                    sound("lose") shake(6, 0.4) burst(16, "7dd3fc") gameOver("-- SWEPT AWAY --") } } } } } }
        Entity { id("log") template(true) startPosition(0, 0) color("a16207") Type { rectangle(90, 70) } pushable(true)
            Events { OnGridStep { Actions { sound("thud") } } } }
        Entity { id("hero") template(true) startPosition(0, 0) color("78350f") Type { circle(40) }
            Movement { GridMove(100, 0.2) { direction(0, 0) } }
            Events { OnCollision("lodge") { Actions { sound("win") shake(5, 0.4) burst(24, "fde68a") nextLevel() } } } }

        # Boulders on the near bank, and a far bank that only opens in the sixth column.
        Entity { id("rockA") startPosition(300, 500) color("57534e") Type { rectangle(100, 100) } solid(true) }
        Entity { id("rockB") startPosition(100, 600) color("57534e") Type { rectangle(100, 100) } solid(true) }
        Entity { id("rockC") startPosition(600, 700) color("57534e") Type { rectangle(100, 100) } solid(true) }
        Entity { id("bankW") startPosition(0, 1000) color("365314") Type { rectangle(500, 100) } solid(true) }
        Entity { id("bankE") startPosition(600, 1000) color("365314") Type { rectangle(200, 100) } solid(true) }
        Entity { id("lodge") startPosition(700, 1100) color("713f12") Type { polygon(6, 50) } }

        Entity { id("croc") startPosition(0, 915) color("16a34a") Type { text("🐊", 56) } facing("left")
            hostile("hero", "-- SNAPPED UP --", "lose")
            Movement { speed(1.5) Path { waypoint(0, 915) waypoint(720, 915) dwell(0.5) } } }

        Entity { id("riverbed") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { OnElapsedTime { Actions {
                spawn("water", 0, 800) spawn("water", 100, 800) spawn("water", 200, 800) spawn("water", 300, 800)
                spawn("water", 400, 800) spawn("water", 500, 800) spawn("water", 600, 800) spawn("water", 700, 800)
                spawn("water", 0, 900) spawn("water", 100, 900) spawn("water", 200, 900) spawn("water", 300, 900)
                spawn("water", 400, 900) spawn("water", 500, 900) spawn("water", 600, 900) spawn("water", 700, 900)
                spawn("log", 205, 715) spawn("log", 605, 515)
                spawn("hero", 310, 410) } } } }
    }
}
