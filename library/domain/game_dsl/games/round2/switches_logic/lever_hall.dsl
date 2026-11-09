Window {
    size(8, 10)
    background("0f172a")
    music("bg_ambient")
    gravity(0)
    Bar("off") { initial(0) }

    Entity {
        id("player")
        startPosition(60, 60)
        color("4ade80")
        Type { rectangle(40, 40) }
        Movement { speed(5) Control { movementControl(0) } }
    }

    # zone 1: lever1 turns off the first beam
    Entity {
        id("lever1")
        startPosition(680, 200)
        color("ef4444")
        Type { rectangle(50, 80) }
        var("pulled", 0)
        Events {
            OnCollision("player") {
                Actions {
                    sound("click")
                    whenVar("pulled", "==", 0) { Actions { delay(0.02) { Actions {
                        changeVar("pulled", 1)
                        setColor("4ade80")
                        setEnable("beam1", false)
                        updateBar("off", 1)
                        whenBar("off", "==", 3) { Actions { broadcast("open") } }
                    } } } }
                    whenVar("pulled", "==", 1) { Actions { delay(0.02) { Actions {
                        changeVar("pulled", -1)
                        setColor("ef4444")
                        setEnable("beam1", true)
                        updateBar("off", -1)
                    } } } }
                }
            }
        }
    }
    Entity {
        id("beam1")
        startPosition(0, 380)
        color("f87171")
        Type { rectangle(800, 14) }
        hostile("player", "-- ZAPPED --", "lose")
    }

    # zone 2: a guard walks near lever2
    Entity {
        id("lever2")
        startPosition(60, 520)
        color("ef4444")
        Type { rectangle(50, 80) }
        var("pulled", 0)
        Events {
            OnCollision("player") {
                Actions {
                    sound("click")
                    whenVar("pulled", "==", 0) { Actions { delay(0.02) { Actions {
                        changeVar("pulled", 1)
                        setColor("4ade80")
                        setEnable("beam2", false)
                        updateBar("off", 1)
                        whenBar("off", "==", 3) { Actions { broadcast("open") } }
                    } } } }
                    whenVar("pulled", "==", 1) { Actions { delay(0.02) { Actions {
                        changeVar("pulled", -1)
                        setColor("ef4444")
                        setEnable("beam2", true)
                        updateBar("off", -1)
                    } } } }
                }
            }
        }
    }
    Entity {
        id("beam2")
        startPosition(0, 700)
        color("fb923c")
        Type { rectangle(800, 14) }
        hostile("player", "-- ZAPPED --", "lose")
    }
    Entity {
        id("guard")
        startPosition(100, 560)
        color("f43f5e")
        Type { circle(24) }
        hostile("player", "-- CAUGHT --", "lose")
        Movement { speed(2.4) Path { waypoint(120, 560) waypoint(650, 560) dwell(0.5) } }
    }

    # zone 3: lever3 is in the corner under the shutter
    Entity {
        id("lever3")
        startPosition(680, 760)
        color("ef4444")
        Type { rectangle(50, 80) }
        var("pulled", 0)
        Events {
            OnCollision("player") {
                Actions {
                    sound("click")
                    whenVar("pulled", "==", 0) { Actions { delay(0.02) { Actions {
                        changeVar("pulled", 1)
                        setColor("4ade80")
                        updateBar("off", 1)
                        whenBar("off", "==", 3) { Actions { broadcast("open") } }
                    } } } }
                    whenVar("pulled", "==", 1) { Actions { delay(0.02) { Actions {
                        changeVar("pulled", -1)
                        setColor("ef4444")
                        updateBar("off", -1)
                    } } } }
                }
            }
        }
    }

    Entity {
        id("shutter")
        startPosition(0, 860)
        color("7c3aed")
        Type { rectangle(800, 24) }
        door("open", "clang")
    }
    Entity {
        id("exit")
        startPosition(360, 920)
        color("fbbf24")
        Type { rectangle(80, 60) }
        Events {
            OnCollision("player") {
                Actions { sound("win") burst(20, "fbbf24") gameOver("-- HALL CLEARED --") }
            }
        }
    }
}
