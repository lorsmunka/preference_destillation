# Echo Caves: the rock walls are invisible but solid. Tap anywhere to send an echo
# that lights them up for a moment; echoes are limited, glow-worms give them back.
Window {
    size(8, 10)
    worldSize(16, 16)
    cameraFollow("hero")
    background("020617")
    music("bg_3")
    gravity(0)

    lives(3, "-- LOST IN THE DARK --", "lose")
    Bar("echoes") {
        initial(8)
        OnValue(0) { Actions { broadcast("silent") } }
    }

    Level {
        Entity {
            id("sonar")
            startPosition(0, 0)
            color("00000000")
            Type { rectangle(1600, 1600) }
            Events {
                OnTapped {
                    Actions {
                        whenBar("echoes", ">", 0) {
                            Actions {
                                updateBar("echoes", -1)
                                sound("blip")
                                burst("hero", 20, "38bdf8")
                                setVisible("rockH", true)
                                setVisible("rockV", true)
                                delay(1.2) { Actions { setVisible("rockH", false) setVisible("rockV", false) } }
                            }
                        }
                    }
                }
            }
        }

        Entity {
            id("hint")
            startPosition(60, 200)
            color("7dd3fc")
            Type { text("Tap anywhere to echo. Key is up top.", 26) }
            Events { OnSignal("silent") { Actions { setText("hint", "No echoes left - feel your way") setColor("f87171") } } }
        }

        Entity {
            id("hero")
            startPosition(80, 80)
            color("fde047")
            Type { circle(20) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnCollision("glow") { Actions { sound("sparkle") updateBar("echoes", 1) burst(10, "a3e635") deleteCollider() } }
                OnSignal("silent") { Actions { sound("beep") flash("f87171", 0.5) } }
                OnCollision("hole") { Actions { sound("win2") shake(4, 0.3) burst(20, "fde047") nextLevel() } }
            }
        }
        joystick("hero", 80, 2.8)

        Entity { id("rockH") template(true) color("475569") solid(true) Type { rectangle(400, 40) }
            Events { OnElapsedTime(2.5) { Actions { setVisible(false) } } } }
        Entity { id("rockV") template(true) color("475569") solid(true) Type { rectangle(40, 200) }
            Events { OnElapsedTime(2.5) { Actions { setVisible(false) } } } }

        Entity {
            id("mason")
            startPosition(0, 0)
            color("00000000")
            Type { circle(1) }
            Events {
                OnElapsedTime {
                    Actions {
                        spawn("rockH", 0, 300) spawn("rockH", 400, 300) spawn("rockH", 800, 300)
                        spawn("rockH", 400, 650) spawn("rockH", 800, 650) spawn("rockH", 1200, 650)
                        spawn("rockH", 0, 1000) spawn("rockH", 400, 1000) spawn("rockH", 800, 1000)
                        spawn("rockH", 400, 1350) spawn("rockH", 800, 1350) spawn("rockH", 1200, 1350)
                        spawn("rockV", 800, 340) spawn("rockV", 500, 450)
                        spawn("rockV", 700, 690) spawn("rockV", 1000, 800)
                        spawn("rockV", 600, 1040) spawn("rockV", 900, 1150)
                    }
                }
            }
        }

        Entity { id("door") startPosition(1300, 0) color("b45309") Type { rectangle(40, 300) } door("caveKey", "clang") }
        Entity { id("caveKey") startPosition(1450, 1450) color("facc15") Type { star(5, 26, 12) } key("caveKey", "hero", "pickup") }
        Entity { id("keyWatch") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { OnSignal("caveKey") { Actions { setText("hint", "The door below is open - go back down") } } } }
        Entity { id("hole") startPosition(1450, 100) color("0ea5e9") Type { ring(50, 30) } }

        Entity { id("glow") template(true) color("a3e635") Type { circle(12) } }
        Entity { id("worms") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { OnElapsedTime { Actions { spawn("glow", 1450, 500) spawn("glow", 100, 820) spawn("glow", 1100, 1450) } } } }

        Entity { id("bat1") startPosition(600, 900) color("ffffff") Type { text("🦇", 40) } facing("left")
            Movement { speed(1.3) chase("hero") }
            Events { OnCollision("hero") { Actions {
                sound("hurt") shake(6, 0.3) flash("hero", "ff0000", 0.3)
                updateBar("lives", -1) setPosition("bat1", 600, 900, 0) } } } }
        Entity { id("bat2") startPosition(300, 1150) color("ffffff") Type { text("🦇", 40) } facing("left")
            Movement { speed(1.3) chase("hero") }
            Events { OnCollision("hero") { Actions {
                sound("hurt") shake(6, 0.3) flash("hero", "ff0000", 0.3)
                updateBar("lives", -1) setPosition("bat2", 300, 1150, 0) } } } }
    }

    Level {
        Entity {
            id("sonar")
            startPosition(0, 0)
            color("00000000")
            Type { rectangle(1600, 1600) }
            Events {
                OnTapped {
                    Actions {
                        whenBar("echoes", ">", 0) {
                            Actions {
                                updateBar("echoes", -1)
                                sound("blip")
                                burst("hero", 20, "38bdf8")
                                setVisible("colH", true)
                                setVisible("colV", true)
                                setVisible("crystal", true)
                                delay(1.2) { Actions { setVisible("colH", false) setVisible("colV", false) setVisible("crystal", false) } }
                            }
                        }
                    }
                }
            }
        }

        Entity {
            id("hint")
            startPosition(40, 1540)
            color("7dd3fc")
            Type { text("The crystal only shines in an echo", 26) }
            Events { OnSignal("silent") { Actions { setText("hint", "No echoes left - feel your way") setColor("f87171") } } }
        }

        Entity {
            id("hero")
            startPosition(80, 1450)
            color("fde047")
            Type { circle(20) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnCollision("glow") { Actions { sound("sparkle") updateBar("echoes", 1) burst(10, "a3e635") deleteCollider() } }
                OnSignal("silent") { Actions { sound("beep") flash("f87171", 0.5) } }
                OnSignal("crystal") { Actions { burst(16, "e879f9") } }
            }
        }
        joystick("hero", 80, 2.8)

        Entity { id("colH") template(true) color("57534e") solid(true) Type { rectangle(200, 40) }
            Events { OnElapsedTime(2.5) { Actions { setVisible(false) } } } }
        Entity { id("colV") template(true) color("57534e") solid(true) Type { rectangle(40, 400) }
            Events { OnElapsedTime(2.5) { Actions { setVisible(false) } } } }

        Entity {
            id("mason")
            startPosition(0, 0)
            color("00000000")
            Type { circle(1) }
            Events {
                OnElapsedTime {
                    Actions {
                        spawn("colV", 350, 400) spawn("colV", 350, 800) spawn("colV", 350, 1200)
                        spawn("colV", 750, 0) spawn("colV", 750, 400) spawn("colV", 750, 800)
                        spawn("colV", 1150, 400) spawn("colV", 1150, 800) spawn("colV", 1150, 1200)
                        spawn("colH", 0, 1000) spawn("colH", 150, 600)
                        spawn("colH", 390, 700) spawn("colH", 550, 1000)
                        spawn("colH", 790, 500) spawn("colH", 950, 900)
                    }
                }
            }
        }

        Entity { id("door") startPosition(1190, 1000) color("b45309") Type { rectangle(410, 40) } door("crystal", "clang") }
        Entity { id("crystal") startPosition(1000, 120) color("e879f9") Type { diamond(44, 60) }
            key("crystal", "hero", "pickup")
            Events { OnElapsedTime(2.5) { Actions { setVisible(false) } } } }
        goal("hole", 1450, 1450, 100, 100, "-- OUT OF THE ECHO CAVES --")

        Entity { id("glow") template(true) color("a3e635") Type { circle(12) } }
        Entity { id("worms") startPosition(0, 0) color("00000000") Type { circle(1) }
            Events { OnElapsedTime { Actions { spawn("glow", 100, 200) spawn("glow", 600, 1450) spawn("glow", 1400, 300) } } } }

        Entity { id("bat1") startPosition(550, 1300) color("ffffff") Type { text("🦇", 40) } facing("left")
            Movement { speed(1.4) chase("hero") }
            Events { OnCollision("hero") { Actions {
                sound("hurt") shake(6, 0.3) flash("hero", "ff0000", 0.3)
                updateBar("lives", -1) setPosition("bat1", 550, 1300, 0) } } } }
        Entity { id("bat2") startPosition(950, 300) color("ffffff") Type { text("🦇", 40) } facing("left")
            Movement { speed(1.4) chase("hero") }
            Events { OnCollision("hero") { Actions {
                sound("hurt") shake(6, 0.3) flash("hero", "ff0000", 0.3)
                updateBar("lives", -1) setPosition("bat2", 950, 300, 0) } } } }
        Entity { id("bat3") startPosition(1400, 600) color("ffffff") Type { text("🦇", 40) } facing("left")
            Movement { speed(1.5) chase("hero") }
            Events { OnCollision("hero") { Actions {
                sound("hurt") shake(6, 0.3) flash("hero", "ff0000", 0.3)
                updateBar("lives", -1) setPosition("bat3", 1400, 600, 0) } } } }
    }
}
