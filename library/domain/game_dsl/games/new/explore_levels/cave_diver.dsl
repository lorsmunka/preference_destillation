# Two flooded caves, top to bottom. Air drains every second; bubbles refill it.
Window {
    size(8, 12)
    worldSize(8, 30)
    cameraFollow("diver")
    background("062a3d")
    music("bg_ambient")
    gravity(0)

    Bar("air") {
        initial(30)
        OnValue(8) { Actions { broadcast("lowAir") } }
        OnValue(0) { Actions { sound("lose") shake(8, 0.5) gameOver("-- OUT OF AIR --") } }
    }
    lives(3, "-- THE EELS WIN --", "lose")

    Level {
        Entity { id("tip") startPosition(30, 2950) color("bae6fd")
            Type { text("CAVE 1/2: find the shell key", 30) }
            Events {
                OnSignal("lowAir") { Actions { setText("tip", "LOW AIR! Chase the bubbles") } }
                OnSignal("shellKey") { Actions { setText("tip", "The gate is open - dive!") } }
            } }

        Entity {
            id("diver")
            startPosition(380, 2850)
            color("fde047")
            Type { circle(22) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnElapsedTime(1) { repeatCount(-1) Actions { updateBar("air", -1) } }
                OnSignal("lowAir") { Actions { sound("beep") flash("ef4444", 0.6) } }
                OnSignal("shellKey") { Actions { burst(14, "facc15") } }
                OnCollision("exit") {
                    Actions {
                        sound("win2")
                        burst(20, "67e8f9")
                        updateBar("air", 10)
                        nextLevel()
                    }
                }
            }
        }
        joystick("diver", 80, 3.4)

        Entity { id("ledgeA") startPosition(0, 2500) color("1e3a4c") solid(true) Type { rectangle(540, 40) } }
        Entity { id("ledgeB") startPosition(260, 2100) color("1e3a4c") solid(true) Type { rectangle(540, 40) } }
        Entity { id("pocket") startPosition(560, 1640) color("1e3a4c") solid(true) Type { rectangle(30, 320) } }
        Entity { id("rockL") startPosition(0, 1600) color("1e3a4c") solid(true) Type { rectangle(500, 40) } }
        Entity { id("gate") startPosition(500, 1600) color("0d9488") Type { rectangle(300, 40) } door("shellKey", "clang") }
        Entity { id("ledgeC") startPosition(0, 1100) color("1e3a4c") solid(true) Type { rectangle(560, 40) } }
        Entity { id("ledgeD") startPosition(240, 600) color("1e3a4c") solid(true) Type { rectangle(560, 40) } }

        Entity { id("shell") startPosition(660, 1700) color("facc15")
            Type { star(5, 24, 11) } key("shellKey", "diver", "pickup") }

        Entity { id("exit") startPosition(60, 80) color("67e8f9")
            Type { ring(50, 34) }
            Events { OnElapsedTime(1.5) { repeatCount(-1) Actions { burst(5, "67e8f9") } } } }

        Entity { id("bubble") template(true) color("90bae6fd") Type { circle(14) }
            Movement { speed(1.2) AutoMove { direction(0, 1) } }
            Events {
                OnCollision("diver") { Actions { sound("bubble") updateBar("air", 4) burst(6, "e0f2fe") deleteEntity() } }
                OnElapsedTime(4) { Actions { deleteEntity() } }
            } }
        Entity { id("vent1") startPosition(400, 2540) color("334155") Type { polygon(5, 16) }
            Events { OnElapsedTime(2.5) { repeatCount(-1) Actions { spawnHere("bubble") } } } }
        Entity { id("vent2") startPosition(520, 2140) color("334155") Type { polygon(5, 16) }
            Events { OnElapsedTime(3) { repeatCount(-1) Actions { spawnHere("bubble") } } } }
        Entity { id("vent3") startPosition(300, 1140) color("334155") Type { polygon(5, 16) }
            Events { OnElapsedTime(3) { repeatCount(-1) Actions { spawnHere("bubble") } } } }
        Entity { id("vent4") startPosition(600, 640) color("334155") Type { polygon(5, 16) }
            Events { OnElapsedTime(3.5) { repeatCount(-1) Actions { spawnHere("bubble") } } } }

        Entity { id("eel1") color("a3e635") Type { rectangle(90, 24) }
            Movement { speed(2.2) Path { waypoint(40, 2300) waypoint(700, 2300) } }
            Events { OnCollision("diver") { Actions {
                sound("hurt") shake(6, 0.3) flash("diver", "ff0000", 0.3)
                updateBar("lives", -1) setPosition("diver", 380, 2850, 0) } } } }
        Entity { id("eel2") color("a3e635") Type { rectangle(90, 24) }
            Movement { speed(2.6) Path { waypoint(700, 1350) waypoint(40, 1350) } }
            Events { OnCollision("diver") { Actions {
                sound("hurt") shake(6, 0.3) flash("diver", "ff0000", 0.3)
                updateBar("lives", -1) setPosition("diver", 380, 2850, 0) } } } }
        Entity { id("eel3") color("a3e635") Type { rectangle(90, 24) }
            Movement { speed(3) Path { waypoint(40, 850) waypoint(700, 850) dwell(0.5) } }
            Events { OnCollision("diver") { Actions {
                sound("hurt") shake(6, 0.3) flash("diver", "ff0000", 0.3)
                updateBar("lives", -1) setPosition("diver", 380, 2850, 0) } } } }
    }

    Level {
        Entity { id("tip") startPosition(30, 2950) color("bae6fd")
            Type { text("CAVE 2/2: open the clam", 30) }
            Events {
                OnSignal("lowAir") { Actions { setText("tip", "LOW AIR! Chase the bubbles") } }
                OnSignal("rockfall") { Actions { setText("tip", "The rubble fell - go deeper") } }
            } }

        Entity {
            id("diver")
            startPosition(380, 2850)
            color("fde047")
            Type { circle(22) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnElapsedTime(1) { repeatCount(-1) Actions { updateBar("air", -1) } }
                OnSignal("lowAir") { Actions { sound("beep") flash("ef4444", 0.6) } }
                OnSignal("pearlKey") { Actions { burst(14, "f5f5f4") } }
                OnCollision("exit") {
                    Actions {
                        sound("win")
                        shake(5, 0.4)
                        burst(30, "67e8f9")
                        gameOver("-- YOU FOUND THE SUNKEN GROTTO --")
                    }
                }
            }
        }
        joystick("diver", 80, 3.4)

        Entity { id("ledgeA") startPosition(220, 2550) color("1e3a4c") solid(true) Type { rectangle(580, 40) } }
        Entity { id("ledgeB") startPosition(0, 2150) color("1e3a4c") solid(true) Type { rectangle(580, 40) } }
        Entity { id("rockL") startPosition(0, 1700) color("1e3a4c") solid(true) Type { rectangle(320, 40) } }
        Entity { id("rockR") startPosition(500, 1700) color("1e3a4c") solid(true) Type { rectangle(300, 40) } }
        Entity { id("rubble") startPosition(320, 1700) color("57534e") solid(true) Type { rectangle(180, 40) }
            Events { OnSignal("rockfall") { Actions { sound("thud") shake(9, 0.5) burst(24, "a8a29e") setEnable(false) } } } }
        Entity { id("ledgeC") startPosition(0, 1200) color("1e3a4c") solid(true) Type { rectangle(520, 40) } }
        Entity { id("rockL2") startPosition(0, 700) color("1e3a4c") solid(true) Type { rectangle(520, 40) } }
        Entity { id("gate") startPosition(520, 700) color("0d9488") Type { rectangle(280, 40) } door("pearlKey", "clang") }

        Entity { id("clam") startPosition(40, 1760) color("c084fc") Type { diamond(70, 40) }
            Events { OnCollision("diver") { Actions { setColor("f0abfc") broadcast("rockfall") } } } }
        Entity { id("pearl") startPosition(80, 820) color("f5f5f4")
            Type { circle(18) } key("pearlKey", "diver", "pickup") }

        Entity { id("exit") startPosition(350, 60) color("67e8f9")
            Type { ring(50, 34) }
            Events { OnElapsedTime(1.5) { repeatCount(-1) Actions { burst(5, "67e8f9") } } } }

        Entity { id("bubble") template(true) color("90bae6fd") Type { circle(14) }
            Movement { speed(1.2) AutoMove { direction(0, 1) } }
            Events {
                OnCollision("diver") { Actions { sound("bubble") updateBar("air", 4) burst(6, "e0f2fe") deleteEntity() } }
                OnElapsedTime(4) { Actions { deleteEntity() } }
            } }
        Entity { id("vent1") startPosition(600, 2590) color("334155") Type { polygon(5, 16) }
            Events { OnElapsedTime(3) { repeatCount(-1) Actions { spawnHere("bubble") } } } }
        Entity { id("vent2") startPosition(200, 2190) color("334155") Type { polygon(5, 16) }
            Events { OnElapsedTime(3) { repeatCount(-1) Actions { spawnHere("bubble") } } } }
        Entity { id("vent3") startPosition(400, 1240) color("334155") Type { polygon(5, 16) }
            Events { OnElapsedTime(3.5) { repeatCount(-1) Actions { spawnHere("bubble") } } } }
        Entity { id("vent4") startPosition(300, 740) color("334155") Type { polygon(5, 16) }
            Events { OnElapsedTime(3.5) { repeatCount(-1) Actions { spawnHere("bubble") } } } }

        Entity { id("eel1") color("a3e635") Type { rectangle(90, 24) }
            Movement { speed(2.6) Path { waypoint(700, 2350) waypoint(40, 2350) } }
            Events { OnCollision("diver") { Actions {
                sound("hurt") shake(6, 0.3) flash("diver", "ff0000", 0.3)
                updateBar("lives", -1) setPosition("diver", 380, 2850, 0) } } } }
        Entity { id("eel2") color("a3e635") Type { rectangle(90, 24) }
            Movement { speed(3) Path { waypoint(40, 1450) waypoint(700, 1450) } }
            Events { OnCollision("diver") { Actions {
                sound("hurt") shake(6, 0.3) flash("diver", "ff0000", 0.3)
                updateBar("lives", -1) setPosition("diver", 380, 2850, 0) } } } }
        Entity { id("eel3") color("a3e635") Type { rectangle(90, 24) }
            Movement { speed(3) Path { waypoint(700, 950) waypoint(40, 950) dwell(0.4) } }
            Events { OnCollision("diver") { Actions {
                sound("hurt") shake(6, 0.3) flash("diver", "ff0000", 0.3)
                updateBar("lives", -1) setPosition("diver", 380, 2850, 0) } } } }
        Entity { id("crab") color("f97316") Type { rectangle(70, 40) }
            Movement { speed(2) Path { waypoint(40, 320) waypoint(700, 320) dwell(1) } }
            Events { OnCollision("diver") { Actions {
                sound("hurt") shake(6, 0.3) flash("diver", "ff0000", 0.3)
                updateBar("lives", -1) setPosition("diver", 380, 2850, 0) } } } }
    }
}
