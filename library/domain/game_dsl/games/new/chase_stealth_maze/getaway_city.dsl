Window {
    size(9, 9)
    worldSize(23, 23)
    cameraFollow("getaway")
    background("18181b")
    music("bg_5")
    gravity(0)

    lives(3, "-- BUSTED --", "lose")
    Bar("cash") {
        initial(0)
        OnValue(3) {
            Actions {
                sound("bell")
                broadcast("garageOpen")
            }
        }
    }

    joystick("getaway", 80, 4.2)

    # Police cars roll in from the four corners of the city, faster and faster.
    Wave(24, 6) { Actions { randomAction {
        Actions { spawn("cop", 60, 60) }
        Actions { spawn("cop", 2200, 60) }
        Actions { spawn("cop", 60, 2200) }
        Actions { spawn("cop", 2200, 2200) } } } }
    Wave(40, 4) { Actions { randomAction {
        Actions { spawn("cop", 60, 60) }
        Actions { spawn("cop", 2200, 60) }
        Actions { spawn("cop", 60, 2200) }
        Actions { spawn("cop", 2200, 2200) } } } }
    Wave(60, 3) { Actions { randomAction {
        Actions { spawn("cop", 60, 60) }
        Actions { spawn("cop", 2200, 60) }
        Actions { spawn("cop", 60, 2200) }
        Actions { spawn("cop", 2200, 2200) } } } }

    # ---------- the getaway car ----------
    Entity {
        id("getaway")
        startPosition(770, 300)
        color("facc15")
        Type { rectangle(56, 36) }
        var("inv", 0)
        Movement {
            speed(0)
            Control { movementControl(0) }
        }
        Events {
            OnCollision("cop") {
                Actions {
                    whenVar("inv", "==", 0) {
                        Actions {
                            changeVar("inv", 1)
                            updateBar("lives", -1)
                            sound("hurt")
                            shake(8, 0.4)
                            flash("ff0000", 1.5)
                            delay(1.5) { Actions { changeVar("inv", -1) } }
                        }
                    }
                    burst(16, "3b82f6")
                    deleteCollider()
                }
            }
            OnCollision("garage") {
                Actions {
                    sound("win")
                    shake(6, 0.5)
                    burst(40, "facc15")
                    gameOver("-- LYING LOW IN THE HIDEOUT --")
                }
            }
            OnSignal("garageOpen") { Actions { flash("4ade80", 0.4) burst(14, "4ade80") } }
            OnSignal("decoy") { Actions { flash("a855f7", 0.3) } }
        }
    }

    Entity { id("hint") startPosition(360, 140) color("a1a1aa") Type { text("Grab 3 bags of cash, then hide in the garage", 22) }
        Events { OnSignal("garageOpen") { Actions { setText("hint", "Garage open! Get to the hideout, north-east") setColor("4ade80") } } } }

    # ---------- police ----------
    Entity {
        id("cop")
        template(true)
        color("2563eb")
        Type { rectangle(56, 36) }
        Movement {
            speed(2.4)
            chase("getaway")
        }
        Events {
            OnElapsedTime(0.5) { repeatCount(-1) Actions { flash("ef4444", 0.25) } }
            OnElapsedTime(25) { Actions { burst(8, "60a5fa") deleteEntity() } }
            OnSignal("decoy") {
                Actions {
                    retarget("decoyCar")
                    delay(5) { Actions { retarget("getaway") } }
                }
            }
        }
    }

    # A look-alike car circles the ring road. Tune a police scanner to it and
    # every cop follows it for five seconds.
    Entity {
        id("decoyCar")
        startPosition(60, 60)
        color("a855f7")
        Type { rectangle(56, 36) }
        Movement {
            speed(3)
            Path {
                waypoint(60, 60)
                waypoint(2200, 60)
                waypoint(2200, 2200)
                waypoint(60, 2200)
            }
        }
        Events { OnSignal("decoy") { Actions { flash("ffffff", 5) burst(20, "c084fc") } } }
    }

    Entity { id("scanner1") startPosition(780, 1480) color("c084fc") Type { ring(18) }
        Events { OnCollision("getaway") { Actions { sound("blip") burst(12, "c084fc") broadcast("decoy") deleteEntity() } } } }
    Entity { id("scanner2") startPosition(1480, 780) color("c084fc") Type { ring(18) }
        Events { OnCollision("getaway") { Actions { sound("blip") burst(12, "c084fc") broadcast("decoy") deleteEntity() } } } }

    # Spike strips slide out of the buildings across two streets.
    Entity { id("spikes1") startPosition(500, 1150) color("e5e7eb") Type { rectangle(200, 24) }
        Movement { speed(1.2) Path { waypoint(500, 1150) waypoint(700, 1150) dwell(1.2) } }
        Events { OnCollision("getaway") { Actions { sound("glass") updateBar("lives", -1) burst(12, "e5e7eb") flash("getaway", "ff0000", 0.4) } } } }
    Entity { id("spikes2") startPosition(1150, 500) color("e5e7eb") Type { rectangle(24, 200) }
        Movement { speed(1.2) Path { waypoint(1150, 500) waypoint(1150, 700) dwell(1.2) } }
        Events { OnCollision("getaway") { Actions { sound("glass") updateBar("lives", -1) burst(12, "e5e7eb") flash("getaway", "ff0000", 0.4) } } } }

    # ---------- the city: 500 px blocks, 200 px streets ----------
    Entity { id("block1") startPosition(200, 200) color("3f3f46") solid(true) Type { rectangle(500, 500) } }
    Entity { id("block2") startPosition(900, 200) color("52525b") solid(true) Type { rectangle(500, 500) } }
    Entity { id("block3") startPosition(1600, 200) color("3f3f46") solid(true) Type { rectangle(500, 500) } }
    Entity { id("block4") startPosition(200, 900) color("52525b") solid(true) Type { rectangle(500, 500) } }
    Entity { id("block5") startPosition(900, 900) color("3f3f46") solid(true) Type { rectangle(500, 500) } }
    Entity { id("block6") startPosition(1600, 900) color("52525b") solid(true) Type { rectangle(500, 500) } }
    Entity { id("block7") startPosition(200, 1600) color("3f3f46") solid(true) Type { rectangle(500, 500) } }
    Entity { id("block8") startPosition(900, 1600) color("52525b") solid(true) Type { rectangle(500, 500) } }

    # The hideout: a walled yard whose garage door faces west.
    Entity { id("hideoutS") startPosition(1600, 1600) color("78716c") solid(true) Type { rectangle(500, 30) } }
    Entity { id("hideoutN") startPosition(1600, 2070) color("78716c") solid(true) Type { rectangle(500, 30) } }
    Entity { id("hideoutE") startPosition(2070, 1630) color("78716c") solid(true) Type { rectangle(30, 440) } }
    Entity { id("hideoutW") startPosition(1600, 1830) color("78716c") solid(true) Type { rectangle(30, 240) } }
    Entity { id("garageDoor") startPosition(1600, 1630) color("b45309") Type { rectangle(30, 200) } door("garageOpen", "clang") }
    Entity { id("garage") startPosition(1800, 1750) color("166534") Type { rectangle(140, 140) } }

    # ---------- the cash ----------
    Entity { id("bag1") startPosition(1480, 300) color("22c55e") Type { diamond(40, 40) }
        Events { OnCollision("getaway") { Actions { sound("coin") updateBar("cash", 1) burst(14, "22c55e") deleteEntity() } } } }
    Entity { id("bag2") startPosition(300, 1480) color("22c55e") Type { diamond(40, 40) }
        Events { OnCollision("getaway") { Actions { sound("coin") updateBar("cash", 1) burst(14, "22c55e") deleteEntity() } } } }
    Entity { id("bag3") startPosition(2180, 1000) color("22c55e") Type { diamond(40, 40) }
        Events { OnCollision("getaway") { Actions { sound("coin") updateBar("cash", 1) burst(14, "22c55e") deleteEntity() } } } }
}
