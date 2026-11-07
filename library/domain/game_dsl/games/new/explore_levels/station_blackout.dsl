# Station Blackout: the power is out on two decks. Find three fuses per deck to bring the
# lights back; power opens the bulkheads, wakes the lift and lights the guide arrows.
Window {
    size(8, 12)
    worldSize(20, 20)
    cameraFollow("hero")
    background("0b0f19")
    music("bg_5")
    gravity(0)

    lives(3, "-- STATION LOST --", "lose")
    Bar("fuses") {
        initial(0)
        OnValue(3) { Actions { broadcast("power") } }
        OnValue(6) { Actions { broadcast("power") } }
    }
    Bar("oxygen") {
        initial(90)
        OnValue(15) { Actions { broadcast("lowO2") } }
        OnValue(0) { Actions { sound("death") gameOver("-- THE AIR RAN OUT --") } }
    }

    Level {
        Entity {
            id("hero")
            startPosition(150, 150)
            color("f8fafc")
            Type { circle(20) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnElapsedTime(1) { repeatCount(-1) Actions { updateBar("oxygen", -1) } }
                OnCollision("fuse") { Actions { sound("pickup") updateBar("fuses", 1) burst(14, "facc15") deleteCollider() } }
                OnCollision("o2") { Actions { sound("bubble") updateBar("oxygen", 20) burst(10, "22d3ee") deleteCollider() } }
                OnSignal("power") { Actions { sound("win2") shake(5, 0.6) flash("fde047", 0.5) } }
                OnSignal("lowO2") { Actions { sound("beep") flash("22d3ee", 0.6) } }
                OnSignal("keycard") { Actions { burst(12, "38bdf8") } }
                OnCollision("lift") { Actions { sound("bell") burst(20, "4ade80") nextLevel() } }
            }
        }
        joystick("hero", 80, 3.2)

        Entity { id("term1") startPosition(160, 460) color("4ade80") Type { text("[terminal]", 28) }
            Events {
                OnCollision("hero") { Actions { sound("click") setText("term1", "Fuses: east bay, mess hall, server room (north-east)") } }
                OnSignal("power") { Actions { setText("term1", "POWER ON - lift is in the north-west corner") } } } }
        Entity { id("term2") startPosition(1200, 1150) color("4ade80") Type { text("[terminal]", 28) }
            Events { OnCollision("hero") { Actions { sound("click") setText("term2", "Server room door needs the keycard in this room") } } } }

        Entity { id("wA") startPosition(0, 600) color("334155") solid(true) Type { rectangle(250, 40) } }
        Entity { id("wB") startPosition(400, 600) color("334155") solid(true) Type { rectangle(900, 40) } }
        Entity { id("wC") startPosition(1450, 600) color("334155") solid(true) Type { rectangle(550, 40) } }
        Entity { id("wD") startPosition(0, 1300) color("334155") solid(true) Type { rectangle(500, 40) } }
        Entity { id("wE") startPosition(650, 1300) color("334155") solid(true) Type { rectangle(950, 40) } }
        Entity { id("wF") startPosition(1750, 1300) color("334155") solid(true) Type { rectangle(250, 40) } }
        Entity { id("wG") startPosition(1000, 0) color("334155") solid(true) Type { rectangle(40, 300) } }
        Entity { id("wH") startPosition(1000, 450) color("334155") solid(true) Type { rectangle(40, 450) } }
        Entity { id("wI") startPosition(1000, 1050) color("334155") solid(true) Type { rectangle(40, 550) } }
        Entity { id("wJ") startPosition(1000, 1750) color("334155") solid(true) Type { rectangle(40, 250) } }

        Entity { id("bulk1") startPosition(1300, 600) color("dc2626") solid(true) Type { rectangle(150, 40) }
            Events { OnSignal("power") { Actions { sound("clang") burst(16, "dc2626") setEnable(false) } } } }
        Entity { id("bulk2") startPosition(1000, 1600) color("dc2626") solid(true) Type { rectangle(40, 150) }
            Events { OnSignal("power") { Actions { sound("clang") burst(16, "dc2626") setEnable(false) } } } }
        Entity { id("serverDoor") startPosition(1600, 1300) color("0ea5e9") Type { rectangle(150, 40) } door("keycard", "clang") }
        Entity { id("card") startPosition(1800, 800) color("38bdf8") Type { rectangle(44, 30) } key("keycard", "hero", "pickup") }

        Entity { id("fuse") template(true) color("facc15") Type { cross(40, 40, 12) } }
        Entity { id("o2") template(true) color("22d3ee") Type { circle(18) } }
        Entity { id("stock") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { OnElapsedTime { Actions {
                spawn("fuse", 1800, 150) spawn("fuse", 300, 1000) spawn("fuse", 1700, 1800)
                spawn("o2", 600, 900) spawn("o2", 1500, 300) } } } }

        Entity { id("lift") startPosition(80, 1820) color("4ade80") Type { rectangle(100, 100) }
            Events { OnElapsedTime { Actions { setEnable(false) } }
                OnSignal("power") { Actions { setEnable(true) } } } }
        Entity { id("arrow1") startPosition(860, 1680) color("fde047") Type { triangle(40, "left") }
            Events { OnElapsedTime { Actions { setVisible(false) } } OnSignal("power") { Actions { setVisible(true) } } } }
        Entity { id("arrow2") startPosition(500, 1840) color("fde047") Type { triangle(40, "left") }
            Events { OnElapsedTime { Actions { setVisible(false) } } OnSignal("power") { Actions { setVisible(true) } } } }

        Entity { id("arc") startPosition(500, 1280) color("a78bfa") Type { rectangle(150, 80) }
            Events { OnCollision("hero") { Actions { sound("buzz") shake(6, 0.3) flash("hero", "a78bfa", 0.4) updateBar("lives", -1) } } } }
        Entity { id("breaker") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { OnElapsedTime(3) { repeatCount(-1) Actions { setEnable("arc", false) delay(1.5) { Actions { setEnable("arc", true) } } } } } }

        Entity { id("drone1") color("ef4444") Type { polygon(6, 24) }
            Movement { speed(2.4) Path { waypoint(1100, 200) waypoint(1900, 200) } }
            Events { OnCollision("hero") { Actions { sound("hurt") shake(6, 0.3) flash("hero", "ff0000", 0.3) updateBar("lives", -1) } } } }
        Entity { id("drone2") color("ef4444") Type { polygon(6, 24) }
            Movement { speed(2) Path { waypoint(100, 750) waypoint(850, 750) waypoint(850, 1150) waypoint(100, 1150) } }
            Events { OnCollision("hero") { Actions { sound("hurt") shake(6, 0.3) flash("hero", "ff0000", 0.3) updateBar("lives", -1) } } } }
        Entity { id("drone3") color("ef4444") Type { polygon(6, 24) }
            Movement { speed(2.6) Path { waypoint(1100, 1500) waypoint(1900, 1900) dwell(0.5) } }
            Events { OnCollision("hero") { Actions { sound("hurt") shake(6, 0.3) flash("hero", "ff0000", 0.3) updateBar("lives", -1) } } } }
    }

    Level {
        Entity {
            id("hero")
            startPosition(150, 150)
            color("f8fafc")
            Type { circle(20) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnElapsedTime(1) { repeatCount(-1) Actions { updateBar("oxygen", -1) } }
                OnCollision("fuse") { Actions { sound("pickup") updateBar("fuses", 1) burst(14, "facc15") deleteCollider() } }
                OnCollision("o2") { Actions { sound("bubble") updateBar("oxygen", 20) burst(10, "22d3ee") deleteCollider() } }
                OnSignal("power") { Actions { sound("win2") shake(5, 0.6) flash("fde047", 0.5) } }
                OnSignal("lowO2") { Actions { sound("beep") flash("22d3ee", 0.6) } }
                OnSignal("bridgeCard") { Actions { burst(12, "38bdf8") } }
            }
        }
        joystick("hero", 80, 3.2)

        Entity { id("term1") startPosition(120, 460) color("4ade80") Type { text("[terminal]", 28) }
            Events {
                OnCollision("hero") { Actions { sound("click") setText("term1", "Deck 2: three fuses, one per wing. Bridge card is west.") } }
                OnSignal("power") { Actions { setText("term1", "POWER ON - escape pod bay, south-east") } } } }

        Entity { id("core") startPosition(700, 700) color("1e293b") solid(true) Type { rectangle(600, 600) } }
        Entity { id("sA") startPosition(980, 0) color("334155") solid(true) Type { rectangle(40, 700) } }
        Entity { id("wA") startPosition(0, 980) color("334155") solid(true) Type { rectangle(200, 40) } }
        Entity { id("wB") startPosition(350, 980) color("334155") solid(true) Type { rectangle(350, 40) } }
        Entity { id("nA") startPosition(980, 1300) color("334155") solid(true) Type { rectangle(40, 500) } }
        Entity { id("nB") startPosition(980, 1950) color("334155") solid(true) Type { rectangle(40, 50) } }
        Entity { id("eA") startPosition(1300, 980) color("334155") solid(true) Type { rectangle(300, 40) } }
        Entity { id("eB") startPosition(1750, 980) color("334155") solid(true) Type { rectangle(250, 40) } }
        Entity { id("bayL") startPosition(1650, 0) color("334155") solid(true) Type { rectangle(40, 300) } }
        Entity { id("bayT1") startPosition(1650, 300) color("334155") solid(true) Type { rectangle(150, 40) } }
        Entity { id("bayT2") startPosition(1950, 300) color("334155") solid(true) Type { rectangle(50, 40) } }

        Entity { id("bayDoor") startPosition(1800, 300) color("dc2626") solid(true) Type { rectangle(150, 40) }
            Events { OnSignal("power") { Actions { sound("clang") burst(16, "dc2626") setEnable(false) } } } }
        Entity { id("bridgeDoor") startPosition(980, 1800) color("0ea5e9") Type { rectangle(40, 150) } door("bridgeCard", "clang") }
        Entity { id("card") startPosition(120, 1700) color("38bdf8") Type { rectangle(44, 30) } key("bridgeCard", "hero", "pickup") }

        Entity { id("fuse") template(true) color("facc15") Type { cross(40, 40, 12) } }
        Entity { id("o2") template(true) color("22d3ee") Type { circle(18) } }
        Entity { id("stock") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { OnElapsedTime { Actions {
                spawn("fuse", 500, 1500) spawn("fuse", 1700, 1700) spawn("fuse", 1300, 500)
                spawn("o2", 450, 1150) spawn("o2", 1550, 1200) } } } }

        goal("pod", 1830, 80, 120, 120, "-- ESCAPE POD LAUNCHED --")
        Entity { id("arrow1") startPosition(1500, 700) color("fde047") Type { triangle(40, "down") }
            Events { OnElapsedTime { Actions { setVisible(false) } } OnSignal("power") { Actions { setVisible(true) } } } }
        Entity { id("arrow2") startPosition(1860, 420) color("fde047") Type { triangle(40, "down") }
            Events { OnElapsedTime { Actions { setVisible(false) } } OnSignal("power") { Actions { setVisible(true) } } } }

        Entity { id("arc") startPosition(1600, 960) color("a78bfa") Type { rectangle(150, 80) }
            Events { OnCollision("hero") { Actions { sound("buzz") shake(6, 0.3) flash("hero", "a78bfa", 0.4) updateBar("lives", -1) } } } }
        Entity { id("breaker") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { OnElapsedTime(2.4) { repeatCount(-1) Actions { setEnable("arc", false) delay(1.2) { Actions { setEnable("arc", true) } } } } } }

        Entity { id("drone1") color("ef4444") Type { polygon(6, 24) }
            Movement { speed(2.6) Path { waypoint(100, 1200) waypoint(850, 1200) waypoint(850, 1900) waypoint(100, 1900) } }
            Events { OnCollision("hero") { Actions { sound("hurt") shake(6, 0.3) flash("hero", "ff0000", 0.3) updateBar("lives", -1) } } } }
        Entity { id("drone2") color("ef4444") Type { polygon(6, 24) }
            Movement { speed(2.8) Path { waypoint(1100, 1400) waypoint(1900, 1400) } }
            Events { OnCollision("hero") { Actions { sound("hurt") shake(6, 0.3) flash("hero", "ff0000", 0.3) updateBar("lives", -1) } } } }
        Entity { id("drone3") color("ef4444") Type { polygon(6, 24) }
            Movement { speed(2.2) Path { waypoint(1100, 650) waypoint(1550, 100) dwell(0.6) } }
            Events { OnCollision("hero") { Actions { sound("hurt") shake(6, 0.3) flash("hero", "ff0000", 0.3) updateBar("lives", -1) } } } }
        Entity { id("chaser") startPosition(1500, 1700) color("f97316") Type { diamond(48, 48) }
            Movement { speed(1.2) chase("hero") }
            Events { OnCollision("hero") { Actions { sound("hurt") flash("hero", "ff0000", 0.3) updateBar("lives", -1) setPosition("chaser", 1500, 1700, 0) } } } }
    }
}
