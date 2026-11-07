Window {
    size(8, 12) worldSize(24, 24) cameraFollow("hero")
    background("05031a") music("bg_4") gravity(0)

    lives(3, "-- HULL BREACHED --", "lose")
    Bar("fuel") { initial(30)
        OnValue(10) { Actions { broadcast("lowFuel") } }
        OnValue(0) { Actions { sound("lose") gameOver("-- ADRIFT WITHOUT FUEL --") } } }
    Bar("stars") { initial(0)
        OnValue(6) { Actions { broadcast("charted") } } }

    Entity { id("hero") startPosition(1180, 300) color("ffffff")
        Type { text("🚀", 48) } facing("right")
        Movement { speed(0) Control { movementControl(0) } }
        Events {
            OnSignal("lowFuel") { Actions { sound("beep") flash("f97316", 0.5) } }
            OnSignal("charted") { Actions { sound("win2") shake(4, 0.4) burst(30, "fde047") } } } }
    joystick("hero", 90, 3.5)

    Entity { id("clock") startPosition(0, 0) color("00000000") Type { circle(1) }
        Events {
            OnElapsedTime { Actions { setEnable("dock", false) } }
            OnElapsedTime(0.7) { repeatCount(-1) Actions { updateBar("fuel", -1) } }
            OnSignal("charted") { Actions { setEnable("dock", true) } } } }
    Entity { id("homeTip") startPosition(1000, 200) color("94a3b8")
        Type { text("trace the six stars in order", 28) }
        Events { OnSignal("charted") { Actions { setText("homeTip", "map complete - dock below!") setColor("fde047") } } } }
    goal("dock", 1100, 60, 200, 60, "-- STAR MAP CHARTED --")

    # Each star wakes the next one; a label appears beside a lit star.
    Entity { id("star1") startPosition(1200, 700) color("64748b") Type { star(5, 34, 15) } var("lit", 0)
        Events { OnCollision("hero") { Actions { whenVar("lit", "==", 0) { Actions {
            changeVar("lit", 1) setColor("fde047") sound("sparkle") burst(16, "fde047")
            updateBar("stars", 1) setVisible("name1", true) broadcast("lit1") } } } } } }
    Entity { id("star2") startPosition(1900, 1100) color("64748b") Type { star(5, 34, 15) } var("lit", 0)
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("lit1") { Actions { setEnable(true) flash("ffffff", 0.4) } }
            OnCollision("hero") { Actions { whenVar("lit", "==", 0) { Actions {
                changeVar("lit", 1) setColor("fde047") sound("sparkle") burst(16, "fde047")
                updateBar("stars", 1) setVisible("name2", true) broadcast("lit2") } } } } } }
    Entity { id("star3") startPosition(2100, 1900) color("64748b") Type { star(5, 34, 15) } var("lit", 0)
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("lit2") { Actions { setEnable(true) flash("ffffff", 0.4) } }
            OnCollision("hero") { Actions { whenVar("lit", "==", 0) { Actions {
                changeVar("lit", 1) setColor("fde047") sound("sparkle") burst(16, "fde047")
                updateBar("stars", 1) setVisible("name3", true) broadcast("lit3") } } } } } }
    Entity { id("star4") startPosition(1300, 2200) color("64748b") Type { star(5, 34, 15) } var("lit", 0)
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("lit3") { Actions { setEnable(true) flash("ffffff", 0.4) } }
            OnCollision("hero") { Actions { whenVar("lit", "==", 0) { Actions {
                changeVar("lit", 1) setColor("fde047") sound("sparkle") burst(16, "fde047")
                updateBar("stars", 1) setVisible("name4", true) broadcast("lit4") } } } } } }
    Entity { id("star5") startPosition(500, 1800) color("64748b") Type { star(5, 34, 15) } var("lit", 0)
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("lit4") { Actions { setEnable(true) flash("ffffff", 0.4) } }
            OnCollision("hero") { Actions { whenVar("lit", "==", 0) { Actions {
                changeVar("lit", 1) setColor("fde047") sound("sparkle") burst(16, "fde047")
                updateBar("stars", 1) setVisible("name5", true) broadcast("lit5") } } } } } }
    Entity { id("star6") startPosition(300, 900) color("64748b") Type { star(5, 34, 15) } var("lit", 0)
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("lit5") { Actions { setEnable(true) flash("ffffff", 0.4) } }
            OnCollision("hero") { Actions { whenVar("lit", "==", 0) { Actions {
                changeVar("lit", 1) setColor("fde047") sound("sparkle") burst(16, "fde047")
                updateBar("stars", 1) setVisible("name6", true) } } } } } }

    Entity { id("name1") startPosition(1270, 720) color("fde68a") Type { text("Lantern", 26) }
        Events { OnElapsedTime { Actions { setVisible(false) } } } }
    Entity { id("name2") startPosition(1970, 1120) color("fde68a") Type { text("Twin", 26) }
        Events { OnElapsedTime { Actions { setVisible(false) } } } }
    Entity { id("name3") startPosition(2000, 1980) color("fde68a") Type { text("Anvil", 26) }
        Events { OnElapsedTime { Actions { setVisible(false) } } } }
    Entity { id("name4") startPosition(1370, 2220) color("fde68a") Type { text("Crown", 26) }
        Events { OnElapsedTime { Actions { setVisible(false) } } } }
    Entity { id("name5") startPosition(570, 1820) color("fde68a") Type { text("Heron", 26) }
        Events { OnElapsedTime { Actions { setVisible(false) } } } }
    Entity { id("name6") startPosition(370, 920) color("fde68a") Type { text("Ember", 26) }
        Events { OnElapsedTime { Actions { setVisible(false) } } } }

    Entity { id("cell1") startPosition(1700, 400) color("22c55e") Type { polygon(6, 20) }
        Events { OnCollision("hero") { Actions { sound("coin") updateBar("fuel", 10) burst(10, "22c55e") deleteEntity() } } } }
    Entity { id("cell2") startPosition(2200, 1500) color("22c55e") Type { polygon(6, 20) }
        Events { OnCollision("hero") { Actions { sound("coin") updateBar("fuel", 10) burst(10, "22c55e") deleteEntity() } } } }
    Entity { id("cell3") startPosition(900, 2300) color("22c55e") Type { polygon(6, 20) }
        Events { OnCollision("hero") { Actions { sound("coin") updateBar("fuel", 10) burst(10, "22c55e") deleteEntity() } } } }
    Entity { id("cell4") startPosition(150, 1350) color("22c55e") Type { polygon(6, 20) }
        Events { OnCollision("hero") { Actions { sound("coin") updateBar("fuel", 10) burst(10, "22c55e") deleteEntity() } } } }

    Entity { id("comet1") color("7dd3fc") Type { circle(26) }
        Movement { speed(3.4) Path { waypoint(0, 1500) waypoint(2350, 1300) } }
        Events { OnCollision("hero") { Actions { sound("hurt") shake(7, 0.4) flash("hero", "ff0000", 0.3) burst(14, "7dd3fc") updateBar("lives", -1) } } } }
    Entity { id("comet2") color("7dd3fc") Type { circle(26) }
        Movement { speed(3) Path { waypoint(1600, 2350) waypoint(1800, 450) } }
        Events { OnCollision("hero") { Actions { sound("hurt") shake(7, 0.4) flash("hero", "ff0000", 0.3) burst(14, "7dd3fc") updateBar("lives", -1) } } } }
    Entity { id("comet3") color("38bdf8") Type { circle(30) }
        Movement { speed(2.6) Path { waypoint(400, 500) waypoint(2000, 700) waypoint(1000, 1500) } }
        Events { OnCollision("hero") { Actions { sound("hurt") shake(7, 0.4) flash("hero", "ff0000", 0.3) burst(14, "38bdf8") updateBar("lives", -1) } } } }

    # The moth steals fuel one unit at a time so the empty-tank check still fires.
    Entity { id("moth") startPosition(200, 2200) color("c084fc") Type { diamond(44, 44) }
        Movement { speed(1.6) chase("hero") }
        Events { OnCollision("hero") { Actions {
            sound("buzz") burst(12, "c084fc") flash("hero", "c084fc", 0.3)
            setPosition("moth", random(100, 2300), 2300, 0)
            loopCount(5) updateBar("fuel", -1) } } } }
}
