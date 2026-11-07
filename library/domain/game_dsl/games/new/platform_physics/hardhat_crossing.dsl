# The worker walks on his own. Tap a dashed gap to drop a girder into it (girders carry over
# between sites), and use STOP / GO to time him past the crane hook and the falling rivets.
Window {
    size(10, 8)
    background("fef9c3")
    music("bg_3")
    gravity(800)

    lives(3, "-- SITE CLOSED --", "lose")
    Bar("beams") { initial(8) }

    Level {
        Entity {
            id("hero")
            startPosition(20, 230)
            color("1f2937")
            facing("left")
            Type { text("🚶", 46) }
            Movement {
                speed(1.1)
                AutoMove { direction(1, 0) }
            }
            Events {
                OnSignal("halt") { Actions { sound("click") setVelocity(0, 0) } }
                OnSignal("walk") { Actions { sound("click") setVelocity(110, 0) } }
                OnCollision("pit") {
                    Actions {
                        sound("death")
                        shake(8, 0.5)
                        updateBar("lives", -1)
                        setPosition("hero", 20, 230, 0)
                        setVelocity(110, 0)
                    }
                }
                OnCollision("hook") {
                    Actions {
                        sound("hurt")
                        flash("ff0000", 0.3)
                        burst(14, "ef4444")
                        updateBar("lives", -1)
                        setPosition("hero", 20, 230, 0)
                        setVelocity(110, 0)
                    }
                }
                OnCollision("lift") {
                    Actions {
                        sound("win2")
                        burst(20, "fde047")
                        nextLevel()
                    }
                }
            }
        }

        Entity { id("deck1") startPosition(0, 200) color("64748b") oneWay(true) Type { rectangle(300, 20) } }
        Entity { id("deck2") startPosition(420, 200) color("64748b") oneWay(true) Type { rectangle(280, 20) } }
        Entity { id("deck3") startPosition(820, 200) color("64748b") oneWay(true) Type { rectangle(180, 20) } }
        Entity { id("girder") template(true) startPosition(0, 0) color("f59e0b") oneWay(true) Type { rectangle(120, 20) } }

        Entity { id("slot1") startPosition(300, 200) color("50f59e0b") Type { rectangle(120, 20) }
            Events { OnTapped("slot1") { Actions {
                whenBar("beams", ">", 0) { Actions { sound("clang") spawn("girder", 300, 200) updateBar("beams", -1) burst(10, "f59e0b") deleteEntity() } }
                whenBar("beams", "<=", 0) { Actions { sound("buzz") flash("ef4444", 0.3) } } } } } }
        Entity { id("slot2") startPosition(700, 200) color("50f59e0b") Type { rectangle(120, 20) }
            Events { OnTapped("slot2") { Actions {
                whenBar("beams", ">", 0) { Actions { sound("clang") spawn("girder", 700, 200) updateBar("beams", -1) burst(10, "f59e0b") deleteEntity() } }
                whenBar("beams", "<=", 0) { Actions { sound("buzz") flash("ef4444", 0.3) } } } } } }

        Entity { id("hook") startPosition(540, 520) color("475569") Type { triangle(44, "down") }
            Movement { speed(2) Path { waypoint(540, 520) waypoint(540, 226) dwell(1.2) } } }
        Entity { id("lift") startPosition(940, 220) color("16a34a") Type { rectangle(60, 100) } }
        Entity { id("pit") startPosition(0, 0) color("78350f") Type { rectangle(1000, 40) } }

        Entity { id("stopPad") startPosition(20, 700) color("ef4444") Type { rectangle(200, 80) }
            Events { OnTapped("stopPad") { Actions { broadcast("halt") } } } }
        Entity { id("goPad") startPosition(240, 700) color("22c55e") Type { rectangle(200, 80) }
            Events { OnTapped("goPad") { Actions { broadcast("walk") } } } }
        Entity { id("stopLbl") startPosition(80, 722) color("ffffff") Type { text("STOP", 32) } }
        Entity { id("goLbl") startPosition(315, 722) color("ffffff") Type { text("GO", 32) } }
        Entity { id("siteLbl") startPosition(800, 740) color("a16207") Type { text("SITE 1/3", 28) } }
    }

    Level {
        Entity {
            id("hero")
            startPosition(20, 230)
            color("1f2937")
            facing("left")
            Type { text("🚶", 46) }
            Movement {
                speed(1.1)
                AutoMove { direction(1, 0) }
            }
            Events {
                OnSignal("halt") { Actions { sound("click") setVelocity(0, 0) } }
                OnSignal("walk") { Actions { sound("click") setVelocity(110, 0) } }
                OnCollision("pit") {
                    Actions {
                        sound("death")
                        shake(8, 0.5)
                        updateBar("lives", -1)
                        setPosition("hero", 20, 230, 0)
                        setVelocity(110, 0)
                    }
                }
                OnCollision("rivet") {
                    Actions {
                        sound("hurt")
                        flash("ff0000", 0.3)
                        shake(5, 0.3)
                        updateBar("lives", -1)
                        deleteCollider()
                    }
                }
                OnCollision("lift") {
                    Actions {
                        sound("win2")
                        burst(20, "fde047")
                        nextLevel()
                    }
                }
            }
        }

        Entity { id("deck1") startPosition(0, 200) color("64748b") oneWay(true) Type { rectangle(220, 20) } }
        Entity { id("deck2") startPosition(340, 200) color("64748b") oneWay(true) Type { rectangle(200, 20) } }
        Entity { id("deck3") startPosition(660, 200) color("64748b") oneWay(true) Type { rectangle(100, 20) } }
        Entity { id("deck4") startPosition(880, 200) color("64748b") oneWay(true) Type { rectangle(120, 20) } }
        Entity { id("girder") template(true) startPosition(0, 0) color("f59e0b") oneWay(true) Type { rectangle(120, 20) } }

        Entity { id("slot1") startPosition(220, 200) color("50f59e0b") Type { rectangle(120, 20) }
            Events { OnTapped("slot1") { Actions {
                whenBar("beams", ">", 0) { Actions { sound("clang") spawn("girder", 220, 200) updateBar("beams", -1) burst(10, "f59e0b") deleteEntity() } }
                whenBar("beams", "<=", 0) { Actions { sound("buzz") flash("ef4444", 0.3) } } } } } }
        Entity { id("slot2") startPosition(540, 200) color("50f59e0b") Type { rectangle(120, 20) }
            Events { OnTapped("slot2") { Actions {
                whenBar("beams", ">", 0) { Actions { sound("clang") spawn("girder", 540, 200) updateBar("beams", -1) burst(10, "f59e0b") deleteEntity() } }
                whenBar("beams", "<=", 0) { Actions { sound("buzz") flash("ef4444", 0.3) } } } } } }
        Entity { id("slot3") startPosition(760, 200) color("50f59e0b") Type { rectangle(120, 20) }
            Events { OnTapped("slot3") { Actions {
                whenBar("beams", ">", 0) { Actions { sound("clang") spawn("girder", 760, 200) updateBar("beams", -1) burst(10, "f59e0b") deleteEntity() } }
                whenBar("beams", "<=", 0) { Actions { sound("buzz") flash("ef4444", 0.3) } } } } } }

        # a spare hard hat on the second deck is worth two girders
        Entity { id("hardhat") startPosition(450, 222) color("facc15") Type { text("⛑", 34) }
            Events { OnCollision("hero") { Actions { sound("pickup") updateBar("beams", 2) burst(12, "facc15") deleteEntity() } } } }

        # the crane trolley drops rivets that ignore gravity and fall at a steady speed
        Entity { id("rivet") template(true) startPosition(0, 0) color("334155") gravity(0) Type { circle(10) }
            Movement { speed(3.2) AutoMove { direction(0, -1) } }
            Events { OnCollision("pit") { Actions { deleteEntity() } } } }
        Entity { id("trolley") startPosition(100, 620) color("f97316") Type { rectangle(70, 24) }
            Movement { speed(2) Path { waypoint(100, 620) waypoint(860, 620) } }
            Events { OnElapsedTime(1.4) { repeatCount(-1) Actions { spawnHere("rivet") } } } }

        Entity { id("lift") startPosition(940, 220) color("16a34a") Type { rectangle(60, 100) } }
        Entity { id("pit") startPosition(0, 0) color("78350f") Type { rectangle(1000, 40) } }

        Entity { id("stopPad") startPosition(20, 700) color("ef4444") Type { rectangle(200, 80) }
            Events { OnTapped("stopPad") { Actions { broadcast("halt") } } } }
        Entity { id("goPad") startPosition(240, 700) color("22c55e") Type { rectangle(200, 80) }
            Events { OnTapped("goPad") { Actions { broadcast("walk") } } } }
        Entity { id("stopLbl") startPosition(80, 722) color("ffffff") Type { text("STOP", 32) } }
        Entity { id("goLbl") startPosition(315, 722) color("ffffff") Type { text("GO", 32) } }
        Entity { id("siteLbl") startPosition(800, 740) color("a16207") Type { text("SITE 2/3", 28) } }
    }

    Level {
        Entity {
            id("hero")
            startPosition(20, 230)
            color("1f2937")
            facing("left")
            Type { text("🚶", 46) }
            Movement {
                speed(1.1)
                AutoMove { direction(1, 0) }
            }
            Events {
                OnSignal("halt") { Actions { sound("click") setVelocity(0, 0) } }
                OnSignal("walk") { Actions { sound("click") setVelocity(110, 0) } }
                OnCollision("pit") {
                    Actions {
                        sound("death")
                        shake(8, 0.5)
                        updateBar("lives", -1)
                        setPosition("hero", 20, 230, 0)
                        setVelocity(110, 0)
                    }
                }
                OnCollision("hook") {
                    Actions {
                        sound("hurt")
                        flash("ff0000", 0.3)
                        burst(14, "ef4444")
                        updateBar("lives", -1)
                        setPosition("hero", 20, 230, 0)
                        setVelocity(110, 0)
                    }
                }
                OnCollision("rivet") {
                    Actions {
                        sound("hurt")
                        flash("ff0000", 0.3)
                        shake(5, 0.3)
                        updateBar("lives", -1)
                        deleteCollider()
                    }
                }
            }
        }

        Entity { id("deck1") startPosition(0, 200) color("64748b") oneWay(true) Type { rectangle(240, 20) } }
        Entity { id("deck2") startPosition(360, 200) color("64748b") oneWay(true) Type { rectangle(160, 20) } }
        Entity { id("deck3") startPosition(640, 200) color("64748b") oneWay(true) Type { rectangle(120, 20) } }
        Entity { id("deck4") startPosition(880, 200) color("64748b") oneWay(true) Type { rectangle(120, 20) } }
        Entity { id("girder") template(true) startPosition(0, 0) color("f59e0b") oneWay(true) Type { rectangle(120, 20) } }

        Entity { id("slot1") startPosition(240, 200) color("50f59e0b") Type { rectangle(120, 20) }
            Events { OnTapped("slot1") { Actions {
                whenBar("beams", ">", 0) { Actions { sound("clang") spawn("girder", 240, 200) updateBar("beams", -1) burst(10, "f59e0b") deleteEntity() } }
                whenBar("beams", "<=", 0) { Actions { sound("buzz") flash("ef4444", 0.3) } } } } } }
        Entity { id("slot2") startPosition(520, 200) color("50f59e0b") Type { rectangle(120, 20) }
            Events { OnTapped("slot2") { Actions {
                whenBar("beams", ">", 0) { Actions { sound("clang") spawn("girder", 520, 200) updateBar("beams", -1) burst(10, "f59e0b") deleteEntity() } }
                whenBar("beams", "<=", 0) { Actions { sound("buzz") flash("ef4444", 0.3) } } } } } }
        Entity { id("slot3") startPosition(760, 200) color("50f59e0b") Type { rectangle(120, 20) }
            Events { OnTapped("slot3") { Actions {
                whenBar("beams", ">", 0) { Actions { sound("clang") spawn("girder", 760, 200) updateBar("beams", -1) burst(10, "f59e0b") deleteEntity() } }
                whenBar("beams", "<=", 0) { Actions { sound("buzz") flash("ef4444", 0.3) } } } } } }

        Entity { id("hook") startPosition(410, 520) color("475569") Type { triangle(44, "down") }
            Movement { speed(2.4) Path { waypoint(410, 520) waypoint(410, 226) dwell(1) } } }
        Entity { id("rivet") template(true) startPosition(0, 0) color("334155") gravity(0) Type { circle(10) }
            Movement { speed(3.6) AutoMove { direction(0, -1) } }
            Events { OnCollision("pit") { Actions { deleteEntity() } } } }
        Entity { id("trolley") startPosition(560, 620) color("f97316") Type { rectangle(70, 24) }
            Movement { speed(2.6) Path { waypoint(560, 620) waypoint(860, 620) } }
            Events { OnElapsedTime(1.1) { repeatCount(-1) Actions { spawnHere("rivet") } } } }

        goal("office", 930, 220, 70, 100, "-- SHIFT COMPLETE --")
        Entity { id("pit") startPosition(0, 0) color("78350f") Type { rectangle(1000, 40) } }

        Entity { id("stopPad") startPosition(20, 700) color("ef4444") Type { rectangle(200, 80) }
            Events { OnTapped("stopPad") { Actions { broadcast("halt") } } } }
        Entity { id("goPad") startPosition(240, 700) color("22c55e") Type { rectangle(200, 80) }
            Events { OnTapped("goPad") { Actions { broadcast("walk") } } } }
        Entity { id("stopLbl") startPosition(80, 722) color("ffffff") Type { text("STOP", 32) } }
        Entity { id("goLbl") startPosition(315, 722) color("ffffff") Type { text("GO", 32) } }
        Entity { id("siteLbl") startPosition(800, 740) color("a16207") Type { text("SITE 3/3", 28) } }
    }
}
