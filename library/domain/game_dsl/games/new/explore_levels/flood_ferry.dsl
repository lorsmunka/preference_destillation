# Flood Ferry: three flooded districts. Pick up everyone before the water takes them;
# the floodgate to the next district only opens once nobody is left waiting.
Window {
    size(9, 9)
    worldSize(27, 12)
    cameraFollow("ferry")
    background("1e3a8a")
    music("bg_5")
    gravity(0)

    lives(4, "-- THE FERRY SANK --", "lose")
    Bar("saved") {
        initial(0)
        OnValue(5) { Actions { sound("bell") } }
    }

    Level {
        Entity { id("ferry") startPosition(100, 560) color("ffffff")
            Type { text("🚤", 50) } facing("left")
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnSignal("allSaved") { Actions { sound("bell") burst(18, "fde047") } }
                OnCollision("exit") { Actions { sound("win2") flash("4ade80", 0.4) burst(20, "4ade80") nextLevel() } }
            } }
        joystick("ferry", 90, 3.6)

        Entity { id("harbor") startPosition(0, 0) color("00000000") Type { circle(1) } var("left", 3)
            Events {
                OnSignal("rescued") { Actions { changeVar("left", -1) broadcast("tally") } }
                OnSignal("swept") { Actions { changeVar("left", -1) broadcast("tally") } }
                OnSignal("tally") { Actions {
                    whenVar("left", "==", 2) { Actions { setText("gateSign", "GATE: 2 still stranded") } }
                    whenVar("left", "==", 1) { Actions { setText("gateSign", "GATE: 1 still stranded") } }
                    whenVar("left", "==", 0) { Actions { setText("gateSign", "GATE OPEN") broadcast("allSaved") } } } }
            } }
        Entity { id("gateSign") startPosition(2050, 900) color("fde68a") Type { text("GATE: 3 still stranded", 28) } }
        Entity { id("title") startPosition(60, 700) color("bfdbfe") Type { text("DISTRICT 1 - MILL ROW", 30) } }

        Entity { id("b1") startPosition(450, 700) color("78716c") solid(true) Type { rectangle(240, 300) } }
        Entity { id("b2") startPosition(500, 100) color("78716c") solid(true) Type { rectangle(200, 300) } }
        Entity { id("b3") startPosition(1000, 450) color("57534e") solid(true) Type { rectangle(260, 260) } }
        Entity { id("b4") startPosition(1500, 0) color("78716c") solid(true) Type { rectangle(220, 420) } }
        Entity { id("b5") startPosition(1550, 800) color("57534e") solid(true) Type { rectangle(260, 400) } }
        Entity { id("b6") startPosition(2000, 350) color("78716c") solid(true) Type { rectangle(200, 500) } }

        Entity { id("s1") startPosition(700, 820) color("ffffff") Type { text("🙋", 40) }
            Events {
                OnCollision("ferry") { Actions { sound("pickup") burst(12, "fde047") updateBar("saved", 1) broadcast("rescued") deleteEntity() } }
                OnElapsedTime(22) { Actions { sound("buzz") flash("ef4444", 8) } }
                OnElapsedTime(30) { Actions { sound("death") burst(16, "60a5fa") updateBar("lives", -1) broadcast("swept") deleteEntity() } }
            } }
        Entity { id("s2") startPosition(1730, 200) color("ffffff") Type { text("🙋", 40) }
            Events {
                OnCollision("ferry") { Actions { sound("pickup") burst(12, "fde047") updateBar("saved", 1) broadcast("rescued") deleteEntity() } }
                OnElapsedTime(32) { Actions { sound("buzz") flash("ef4444", 8) } }
                OnElapsedTime(40) { Actions { sound("death") burst(16, "60a5fa") updateBar("lives", -1) broadcast("swept") deleteEntity() } }
            } }
        Entity { id("s3") startPosition(2210, 700) color("ffffff") Type { text("🙋", 40) }
            Events {
                OnCollision("ferry") { Actions { sound("pickup") burst(12, "fde047") updateBar("saved", 1) broadcast("rescued") deleteEntity() } }
                OnElapsedTime(42) { Actions { sound("buzz") flash("ef4444", 8) } }
                OnElapsedTime(50) { Actions { sound("death") burst(16, "60a5fa") updateBar("lives", -1) broadcast("swept") deleteEntity() } }
            } }

        Entity { id("log1") color("92400e") Type { rectangle(40, 130) }
            Movement { speed(2.5) Path { waypoint(880, 40) waypoint(880, 1060) } }
            Events { OnCollision("ferry") { Actions { sound("hurt") shake(6, 0.3) flash("ferry", "ff0000", 0.3) updateBar("lives", -1) } } } }
        Entity { id("log2") color("92400e") Type { rectangle(40, 130) }
            Movement { speed(3) Path { waypoint(1360, 1060) waypoint(1360, 40) } }
            Events { OnCollision("ferry") { Actions { sound("hurt") shake(6, 0.3) flash("ferry", "ff0000", 0.3) updateBar("lives", -1) } } } }
        Entity { id("pool1") startPosition(1240, 880) color("93c5fd") Type { ring(44, 26) }
            Events { OnCollision("ferry") { Actions { sound("bubble") shake(4, 0.4) setPosition("ferry", 100, 560, 0.8) } } } }

        Entity { id("wallLow") startPosition(2400, 0) color("334155") solid(true) Type { rectangle(40, 500) } }
        Entity { id("wallHigh") startPosition(2400, 700) color("334155") solid(true) Type { rectangle(40, 500) } }
        Entity { id("gate") startPosition(2400, 500) color("f59e0b") Type { rectangle(40, 200) } door("allSaved", "clang") }
        Entity { id("exit") startPosition(2560, 540) color("4ade80") Type { ring(50, 30) } }
    }

    Level {
        Entity { id("ferry") startPosition(100, 1000) color("ffffff")
            Type { text("🚤", 50) } facing("left")
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnSignal("allSaved") { Actions { sound("bell") burst(18, "fde047") } }
                OnCollision("exit") { Actions { sound("win2") flash("4ade80", 0.4) burst(20, "4ade80") nextLevel() } }
            } }
        joystick("ferry", 90, 3.6)

        Entity { id("harbor") startPosition(0, 0) color("00000000") Type { circle(1) } var("left", 3)
            Events {
                OnSignal("rescued") { Actions { changeVar("left", -1) broadcast("tally") } }
                OnSignal("swept") { Actions { changeVar("left", -1) broadcast("tally") } }
                OnSignal("tally") { Actions {
                    whenVar("left", "==", 2) { Actions { setText("gateSign", "GATE: 2 still stranded") } }
                    whenVar("left", "==", 1) { Actions { setText("gateSign", "GATE: 1 still stranded") } }
                    whenVar("left", "==", 0) { Actions { setText("gateSign", "GATE OPEN") broadcast("allSaved") } } } }
            } }
        Entity { id("gateSign") startPosition(2020, 120) color("fde68a") Type { text("GATE: 3 still stranded", 28) } }
        Entity { id("title") startPosition(60, 1120) color("bfdbfe") Type { text("DISTRICT 2 - MARKET SQUARE", 30) } }

        Entity { id("b1") startPosition(350, 400) color("78716c") solid(true) Type { rectangle(300, 500) } }
        Entity { id("b2") startPosition(900, 0) color("57534e") solid(true) Type { rectangle(240, 600) } }
        Entity { id("b3") startPosition(900, 850) color("78716c") solid(true) Type { rectangle(240, 350) } }
        Entity { id("b4") startPosition(1450, 300) color("57534e") solid(true) Type { rectangle(300, 300) } }
        Entity { id("b5") startPosition(1950, 650) color("78716c") solid(true) Type { rectangle(240, 550) } }

        Entity { id("s1") startPosition(240, 160) color("ffffff") Type { text("🙋", 40) }
            Events {
                OnCollision("ferry") { Actions { sound("pickup") burst(12, "fde047") updateBar("saved", 1) broadcast("rescued") deleteEntity() } }
                OnElapsedTime(20) { Actions { sound("buzz") flash("ef4444", 8) } }
                OnElapsedTime(28) { Actions { sound("death") burst(16, "60a5fa") updateBar("lives", -1) broadcast("swept") deleteEntity() } }
            } }
        Entity { id("s2") startPosition(1530, 620) color("ffffff") Type { text("🙋", 40) }
            Events {
                OnCollision("ferry") { Actions { sound("pickup") burst(12, "fde047") updateBar("saved", 1) broadcast("rescued") deleteEntity() } }
                OnElapsedTime(28) { Actions { sound("buzz") flash("ef4444", 8) } }
                OnElapsedTime(36) { Actions { sound("death") burst(16, "60a5fa") updateBar("lives", -1) broadcast("swept") deleteEntity() } }
            } }
        Entity { id("s3") startPosition(2210, 1080) color("ffffff") Type { text("🙋", 40) }
            Events {
                OnCollision("ferry") { Actions { sound("pickup") burst(12, "fde047") updateBar("saved", 1) broadcast("rescued") deleteEntity() } }
                OnElapsedTime(38) { Actions { sound("buzz") flash("ef4444", 8) } }
                OnElapsedTime(46) { Actions { sound("death") burst(16, "60a5fa") updateBar("lives", -1) broadcast("swept") deleteEntity() } }
            } }

        Entity { id("log1") color("92400e") Type { rectangle(130, 40) }
            Movement { speed(3) Path { waypoint(40, 280) waypoint(800, 280) } }
            Events { OnCollision("ferry") { Actions { sound("hurt") shake(6, 0.3) flash("ferry", "ff0000", 0.3) updateBar("lives", -1) } } } }
        Entity { id("log2") color("92400e") Type { rectangle(40, 130) }
            Movement { speed(3.2) Path { waypoint(1300, 40) waypoint(1300, 1060) } }
            Events { OnCollision("ferry") { Actions { sound("hurt") shake(6, 0.3) flash("ferry", "ff0000", 0.3) updateBar("lives", -1) } } } }
        Entity { id("log3") color("92400e") Type { rectangle(40, 130) }
            Movement { speed(2.6) Path { waypoint(1850, 1060) waypoint(1850, 40) dwell(0.5) } }
            Events { OnCollision("ferry") { Actions { sound("hurt") shake(6, 0.3) flash("ferry", "ff0000", 0.3) updateBar("lives", -1) } } } }
        Entity { id("pool1") startPosition(700, 700) color("93c5fd") Type { ring(44, 26) }
            Events { OnCollision("ferry") { Actions { sound("bubble") shake(4, 0.4) setPosition("ferry", 100, 1000, 0.8) } } } }
        Entity { id("pool2") startPosition(1550, 900) color("93c5fd") Type { ring(44, 26) }
            Events { OnCollision("ferry") { Actions { sound("bubble") shake(4, 0.4) setPosition("ferry", 100, 1000, 0.8) } } } }

        Entity { id("wallLow") startPosition(2400, 0) color("334155") solid(true) Type { rectangle(40, 250) } }
        Entity { id("wallHigh") startPosition(2400, 450) color("334155") solid(true) Type { rectangle(40, 750) } }
        Entity { id("gate") startPosition(2400, 250) color("f59e0b") Type { rectangle(40, 200) } door("allSaved", "clang") }
        Entity { id("exit") startPosition(2560, 290) color("4ade80") Type { ring(50, 30) } }
    }

    Level {
        Entity { id("ferry") startPosition(100, 120) color("ffffff")
            Type { text("🚤", 50) } facing("left")
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnSignal("allSaved") { Actions { sound("bell") burst(18, "fde047") } }
                OnCollision("exit") { Actions { sound("win") shake(6, 0.5) burst(30, "4ade80") gameOver("-- THE CITY IS EVACUATED --") } }
            } }
        joystick("ferry", 90, 3.6)

        Entity { id("harbor") startPosition(0, 0) color("00000000") Type { circle(1) } var("left", 4)
            Events {
                OnSignal("rescued") { Actions { changeVar("left", -1) broadcast("tally") } }
                OnSignal("swept") { Actions { changeVar("left", -1) broadcast("tally") } }
                OnSignal("tally") { Actions {
                    whenVar("left", "==", 3) { Actions { setText("gateSign", "LAST GATE: 3 stranded") } }
                    whenVar("left", "==", 2) { Actions { setText("gateSign", "LAST GATE: 2 stranded") } }
                    whenVar("left", "==", 1) { Actions { setText("gateSign", "LAST GATE: 1 stranded") } }
                    whenVar("left", "==", 0) { Actions { setText("gateSign", "LAST GATE OPEN") broadcast("allSaved") } } } }
            } }
        Entity { id("gateSign") startPosition(2020, 1000) color("fde68a") Type { text("LAST GATE: 4 stranded", 28) } }
        Entity { id("title") startPosition(60, 300) color("bfdbfe") Type { text("DISTRICT 3 - THE DOCKS", 30) } }

        Entity { id("b1") startPosition(300, 0) color("78716c") solid(true) Type { rectangle(200, 700) } }
        Entity { id("b2") startPosition(750, 450) color("57534e") solid(true) Type { rectangle(240, 750) } }
        Entity { id("b3") startPosition(1250, 0) color("78716c") solid(true) Type { rectangle(240, 650) } }
        Entity { id("b4") startPosition(1700, 400) color("57534e") solid(true) Type { rectangle(260, 800) } }

        Entity { id("s1") startPosition(520, 600) color("ffffff") Type { text("🙋", 40) }
            Events {
                OnCollision("ferry") { Actions { sound("pickup") burst(12, "fde047") updateBar("saved", 1) broadcast("rescued") deleteEntity() } }
                OnElapsedTime(16) { Actions { sound("buzz") flash("ef4444", 8) } }
                OnElapsedTime(24) { Actions { sound("death") burst(16, "60a5fa") updateBar("lives", -1) broadcast("swept") deleteEntity() } }
            } }
        Entity { id("s2") startPosition(600, 1100) color("ffffff") Type { text("🙋", 40) }
            Events {
                OnCollision("ferry") { Actions { sound("pickup") burst(12, "fde047") updateBar("saved", 1) broadcast("rescued") deleteEntity() } }
                OnElapsedTime(24) { Actions { sound("buzz") flash("ef4444", 8) } }
                OnElapsedTime(32) { Actions { sound("death") burst(16, "60a5fa") updateBar("lives", -1) broadcast("swept") deleteEntity() } }
            } }
        Entity { id("s3") startPosition(1510, 560) color("ffffff") Type { text("🙋", 40) }
            Events {
                OnCollision("ferry") { Actions { sound("pickup") burst(12, "fde047") updateBar("saved", 1) broadcast("rescued") deleteEntity() } }
                OnElapsedTime(32) { Actions { sound("buzz") flash("ef4444", 8) } }
                OnElapsedTime(40) { Actions { sound("death") burst(16, "60a5fa") updateBar("lives", -1) broadcast("swept") deleteEntity() } }
            } }
        Entity { id("s4") startPosition(1980, 200) color("ffffff") Type { text("🙋", 40) }
            Events {
                OnCollision("ferry") { Actions { sound("pickup") burst(12, "fde047") updateBar("saved", 1) broadcast("rescued") deleteEntity() } }
                OnElapsedTime(40) { Actions { sound("buzz") flash("ef4444", 8) } }
                OnElapsedTime(48) { Actions { sound("death") burst(16, "60a5fa") updateBar("lives", -1) broadcast("swept") deleteEntity() } }
            } }

        Entity { id("log1") color("92400e") Type { rectangle(40, 130) }
            Movement { speed(3) Path { waypoint(620, 40) waypoint(620, 1060) } }
            Events { OnCollision("ferry") { Actions { sound("hurt") shake(6, 0.3) flash("ferry", "ff0000", 0.3) updateBar("lives", -1) } } } }
        Entity { id("log2") color("92400e") Type { rectangle(130, 40) }
            Movement { speed(3.4) Path { waypoint(1000, 1000) waypoint(1600, 1000) waypoint(1600, 800) waypoint(1000, 800) } }
            Events { OnCollision("ferry") { Actions { sound("hurt") shake(6, 0.3) flash("ferry", "ff0000", 0.3) updateBar("lives", -1) } } } }
        Entity { id("log3") color("92400e") Type { rectangle(40, 130) }
            Movement { speed(3.6) Path { waypoint(2150, 40) waypoint(2150, 1060) } }
            Events { OnCollision("ferry") { Actions { sound("hurt") shake(6, 0.3) flash("ferry", "ff0000", 0.3) updateBar("lives", -1) } } } }
        Entity { id("pool1") startPosition(1080, 200) color("93c5fd") Type { ring(44, 26) }
            Events { OnCollision("ferry") { Actions { sound("bubble") shake(4, 0.4) setPosition("ferry", 100, 120, 0.8) } } } }

        Entity { id("wallLow") startPosition(2400, 0) color("334155") solid(true) Type { rectangle(40, 800) } }
        Entity { id("wallHigh") startPosition(2400, 1000) color("334155") solid(true) Type { rectangle(40, 200) } }
        Entity { id("gate") startPosition(2400, 800) color("f59e0b") Type { rectangle(40, 200) } door("allSaved", "clang") }
        Entity { id("exit") startPosition(2560, 840) color("4ade80") Type { ring(50, 30) } }
    }
}
