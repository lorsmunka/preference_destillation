# Lantern Rooftops: swipe to leap from roof to roof and light the six street lamps.
# A fall into an alley costs a life and puts you back on the roof you jumped from.
Window {
    size(8, 10)
    worldSize(36, 10)
    cameraFollow("hero")
    background("111827")
    music("bg_2")
    gravity(1200)

    lives(4, "-- TOO MANY FALLS --", "lose")
    Bar("lamps") {
        initial(0)
        OnValue(6) { Actions { broadcast("allLit") } }
    }

    Entity {
        id("gesture")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        var("ready", 1)
        Events {
            OnSwipe("right") { Actions { whenVar("ready", "==", 1) { Actions {
                changeVar("ready", -1) setVelocity("hero", 220, 720) sound("jump")
                delay(1) { Actions { changeVar("ready", 1) } } } } } }
            OnSwipe("left") { Actions { whenVar("ready", "==", 1) { Actions {
                changeVar("ready", -1) setVelocity("hero", -220, 720) sound("jump")
                delay(1) { Actions { changeVar("ready", 1) } } } } } }
            OnSwipe("up") { Actions { whenVar("ready", "==", 1) { Actions {
                changeVar("ready", -1) setVelocity("hero", 0, 760) sound("jump")
                delay(1) { Actions { changeVar("ready", 1) } } } } } }
            OnSwipe("down") { Actions { setVelocity("hero", 0, -200) } }
        }
    }

    Entity {
        id("hero")
        startPosition(120, 320)
        color("fbbf24")
        Type { rectangle(34, 46) }
        Movement { speed(0) AutoMove { direction(0, 0) } }
        Events {
            OnSignal("allLit") { Actions { sound("bell") shake(4, 0.4) burst(24, "fde047") } }
        }
    }

    Entity {
        id("hint")
        startPosition(30, 560)
        color("fde68a")
        Type { text("Swipe left, right or up to leap. Light all six lamps.", 24) }
        Events { OnSignal("allLit") { Actions { setText("hint", "Every lamp burns - the bell tower is open!") } } }
    }

    # Roofs: one-way, so you land on top. Each thin strip on a roof stops your slide when you land.
    Entity { id("r1") startPosition(0, 0) color("374151") Type { rectangle(400, 300) } oneWay(true) }
    Entity { id("r2") startPosition(520, 0) color("4b5563") Type { rectangle(280, 380) } oneWay(true) }
    Entity { id("r3") startPosition(950, 0) color("374151") Type { rectangle(300, 300) } oneWay(true) }
    Entity { id("r4") startPosition(1380, 0) color("4b5563") Type { rectangle(320, 440) } oneWay(true) }
    Entity { id("r5") startPosition(1850, 0) color("374151") Type { rectangle(300, 360) } oneWay(true) }
    Entity { id("r6") startPosition(2280, 0) color("4b5563") Type { rectangle(320, 500) } oneWay(true) }
    Entity { id("r7") startPosition(2760, 0) color("374151") Type { rectangle(340, 400) } oneWay(true) }
    Entity { id("r8") startPosition(3220, 0) color("4b5563") Type { rectangle(380, 320) } oneWay(true) }

    Entity { id("land1") startPosition(0, 300) color("00000000") Type { rectangle(400, 10) }
        Events { OnCollision("hero") { Actions { setVelocity("hero", 0, 0) } } } }
    Entity { id("land2") startPosition(520, 380) color("00000000") Type { rectangle(280, 10) }
        Events { OnCollision("hero") { Actions { setVelocity("hero", 0, 0) } } } }
    Entity { id("land3") startPosition(950, 300) color("00000000") Type { rectangle(300, 10) }
        Events { OnCollision("hero") { Actions { setVelocity("hero", 0, 0) } } } }
    Entity { id("land4") startPosition(1380, 440) color("00000000") Type { rectangle(320, 10) }
        Events { OnCollision("hero") { Actions { setVelocity("hero", 0, 0) } } } }
    Entity { id("land5") startPosition(1850, 360) color("00000000") Type { rectangle(300, 10) }
        Events { OnCollision("hero") { Actions { setVelocity("hero", 0, 0) } } } }
    Entity { id("land6") startPosition(2280, 500) color("00000000") Type { rectangle(320, 10) }
        Events { OnCollision("hero") { Actions { setVelocity("hero", 0, 0) } } } }
    Entity { id("land7") startPosition(2760, 400) color("00000000") Type { rectangle(340, 10) }
        Events { OnCollision("hero") { Actions { setVelocity("hero", 0, 0) } } } }
    Entity { id("land8") startPosition(3220, 320) color("00000000") Type { rectangle(380, 10) }
        Events { OnCollision("hero") { Actions { setVelocity("hero", 0, 0) } } } }

    # Alleys: falling in costs a life and puts you back on the roof before the gap.
    Entity { id("alley1") startPosition(400, 0) color("1f2937") Type { rectangle(120, 60) }
        Events { OnCollision("hero") { Actions { sound("thud") shake(6, 0.3) updateBar("lives", -1) setVelocity("hero", 0, 0) setPosition("hero", 300, 320, 0) } } } }
    Entity { id("alley2") startPosition(800, 0) color("1f2937") Type { rectangle(150, 60) }
        Events { OnCollision("hero") { Actions { sound("thud") shake(6, 0.3) updateBar("lives", -1) setVelocity("hero", 0, 0) setPosition("hero", 700, 400, 0) } } } }
    Entity { id("alley3") startPosition(1250, 0) color("1f2937") Type { rectangle(130, 60) }
        Events { OnCollision("hero") { Actions { sound("thud") shake(6, 0.3) updateBar("lives", -1) setVelocity("hero", 0, 0) setPosition("hero", 1200, 320, 0) } } } }
    Entity { id("alley4") startPosition(1700, 0) color("1f2937") Type { rectangle(150, 60) }
        Events { OnCollision("hero") { Actions { sound("thud") shake(6, 0.3) updateBar("lives", -1) setVelocity("hero", 0, 0) setPosition("hero", 1640, 460, 0) } } } }
    Entity { id("alley5") startPosition(2150, 0) color("1f2937") Type { rectangle(130, 60) }
        Events { OnCollision("hero") { Actions { sound("thud") shake(6, 0.3) updateBar("lives", -1) setVelocity("hero", 0, 0) setPosition("hero", 2100, 380, 0) } } } }
    Entity { id("alley6") startPosition(2600, 0) color("1f2937") Type { rectangle(160, 60) }
        Events { OnCollision("hero") { Actions { sound("thud") shake(6, 0.3) updateBar("lives", -1) setVelocity("hero", 0, 0) setPosition("hero", 2500, 520, 0) } } } }
    Entity { id("alley7") startPosition(3100, 0) color("1f2937") Type { rectangle(120, 60) }
        Events { OnCollision("hero") { Actions { sound("thud") shake(6, 0.3) updateBar("lives", -1) setVelocity("hero", 0, 0) setPosition("hero", 3000, 420, 0) } } } }

    Entity { id("lamp1") startPosition(640, 380) color("57534e") Type { rectangle(24, 220) } var("lit", 0)
        Events { OnCollision("hero") { Actions { whenVar("lit", "==", 0) { Actions {
            changeVar("lit", 1) setColor("fde047") sound("sparkle") burst(16, "fde047") updateBar("lamps", 1) } } } } } }
    Entity { id("lamp2") startPosition(1180, 300) color("57534e") Type { rectangle(24, 220) } var("lit", 0)
        Events { OnCollision("hero") { Actions { whenVar("lit", "==", 0) { Actions {
            changeVar("lit", 1) setColor("fde047") sound("sparkle") burst(16, "fde047") updateBar("lamps", 1) } } } } } }
    Entity { id("lamp3") startPosition(1520, 440) color("57534e") Type { rectangle(24, 220) } var("lit", 0)
        Events { OnCollision("hero") { Actions { whenVar("lit", "==", 0) { Actions {
            changeVar("lit", 1) setColor("fde047") sound("sparkle") burst(16, "fde047") updateBar("lamps", 1) } } } } } }
    Entity { id("lamp4") startPosition(1900, 360) color("57534e") Type { rectangle(24, 220) } var("lit", 0)
        Events { OnCollision("hero") { Actions { whenVar("lit", "==", 0) { Actions {
            changeVar("lit", 1) setColor("fde047") sound("sparkle") burst(16, "fde047") updateBar("lamps", 1) } } } } } }
    Entity { id("lamp5") startPosition(2560, 500) color("57534e") Type { rectangle(24, 220) } var("lit", 0)
        Events { OnCollision("hero") { Actions { whenVar("lit", "==", 0) { Actions {
            changeVar("lit", 1) setColor("fde047") sound("sparkle") burst(16, "fde047") updateBar("lamps", 1) } } } } } }
    Entity { id("lamp6") startPosition(2930, 400) color("57534e") Type { rectangle(24, 220) } var("lit", 0)
        Events { OnCollision("hero") { Actions { whenVar("lit", "==", 0) { Actions {
            changeVar("lit", 1) setColor("fde047") sound("sparkle") burst(16, "fde047") updateBar("lamps", 1) } } } } } }

    Entity { id("cat") color("ffffff") Type { text("🐈", 44) } facing("left")
        Movement { speed(1.4) Path { waypoint(1400, 440) waypoint(1540, 440) dwell(0.8) } }
        Events { OnCollision("hero") { Actions { sound("hurt") flash("hero", "ef4444", 0.3) updateBar("lives", -1) setVelocity("hero", -200, 600) } } } }

    Entity { id("smoke") template(true) color("9ca3af") gravity(0) Type { circle(22) }
        Movement { speed(1.1) AutoMove { direction(0, 1) } }
        Events {
            OnCollision("hero") { Actions { sound("hurt") flash("hero", "6b7280", 0.4) updateBar("lives", -1) deleteEntity() } }
            OnElapsedTime(2.6) { Actions { deleteEntity() } } } }
    Entity { id("chimney") startPosition(2400, 500) color("7c2d12") Type { rectangle(44, 70) }
        Events { OnElapsedTime(random(1.2, 2.2)) { repeatCount(-1) Actions { spawn("smoke", 2400, 580) } } } }

    Entity { id("bellSign") startPosition(3260, 520) color("94a3b8") Type { text("BELL TOWER - dark", 26) }
        Events { OnSignal("allLit") { Actions { setText("bellSign", "BELL TOWER - open") setColor("fde047") } } } }
    Entity { id("keeper") startPosition(0, 0) color("00000000") Type { circle(1) }
        Events {
            OnElapsedTime { Actions { setEnable("tower", false) } }
            OnSignal("allLit") { Actions { setEnable("tower", true) } } } }
    goal("tower", 3440, 320, 90, 150, "-- THE CITY IS LIT --")
}
