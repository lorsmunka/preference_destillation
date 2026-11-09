Window {
    size(12, 6)
    background("020617")
    music("bg_6")
    Bar("air") {
        initial(70)
        OnValue(0) { Actions { gameOver("-- OUT OF AIR --") } }
    }

    Entity {
        id("hero")
        startPosition(40, 270)
        color("22d3ee")
        Type { rectangle(36, 36) }
        Movement { speed(4) Control { movementControl(0) } }
        Events {
            OnElapsedTime { Actions { ramp("air", 70, 0, 70) } }
        }
    }

    # first airlock: gate1 | console1 | gate2
    Entity {
        id("gate1")
        startPosition(250, 0)
        color("f59e0b")
        solid(true)
        Type { rectangle(24, 600) }
        Events { OnElapsedTime { Actions { setEnable(false) } } }
    }
    Entity {
        id("gate2")
        startPosition(450, 0)
        color("f59e0b")
        solid(true)
        Type { rectangle(24, 600) }
    }
    Entity {
        id("console1")
        startPosition(310, 500)
        color("fb7185")
        Type { rectangle(90, 60) }
        var("side", 0)
        Events {
            OnTapped("console1") {
                Actions {
                    sound("click")
                    whenVar("side", "==", 0) {
                        Actions { delay(0.02) { Actions {
                            changeVar("side", 1)
                            setEnable("gate1", true)
                            flash("fde047", 1.2)
                            delay(1.2) { Actions { setEnable("gate2", false) sound("bump") } }
                        } } }
                    }
                    whenVar("side", "==", 1) {
                        Actions { delay(0.02) { Actions {
                            changeVar("side", -1)
                            setEnable("gate2", true)
                            flash("fde047", 1.2)
                            delay(1.2) { Actions { setEnable("gate1", false) sound("bump") } }
                        } } }
                    }
                }
            }
        }
    }

    # the corridor between the airlocks
    Entity {
        id("sentry1")
        startPosition(540, 60)
        color("ef4444")
        Type { circle(22) }
        hostile("hero", "-- SENTRY HIT --", "lose")
        Movement { speed(3) Path { waypoint(540, 60) waypoint(540, 540) } }
    }
    Entity {
        id("sentry2")
        startPosition(620, 540)
        color("f97316")
        Type { circle(22) }
        hostile("hero", "-- SENTRY HIT --", "lose")
        Movement { speed(4.5) Path { waypoint(620, 540) waypoint(620, 60) dwell(0.3) } }
    }
    Entity {
        id("sentry3")
        startPosition(500, 300)
        color("e11d48")
        Type { circle(18) }
        hostile("hero", "-- SENTRY HIT --", "lose")
        Movement { speed(2) chase("hero", 1.5) }
    }

    # second airlock: gate3 | console2 | gate4
    Entity {
        id("gate3")
        startPosition(750, 0)
        color("f59e0b")
        solid(true)
        Type { rectangle(24, 600) }
        Events { OnElapsedTime { Actions { setEnable(false) } } }
    }
    Entity {
        id("gate4")
        startPosition(950, 0)
        color("f59e0b")
        solid(true)
        Type { rectangle(24, 600) }
    }
    Entity {
        id("console2")
        startPosition(810, 20)
        color("fb7185")
        Type { rectangle(90, 60) }
        var("side", 0)
        Events {
            OnTapped("console2") {
                Actions {
                    sound("click")
                    whenVar("side", "==", 0) {
                        Actions { delay(0.02) { Actions {
                            changeVar("side", 1)
                            setEnable("gate3", true)
                            flash("fde047", 1.2)
                            delay(1.2) { Actions { setEnable("gate4", false) sound("bump") } }
                        } } }
                    }
                    whenVar("side", "==", 1) {
                        Actions { delay(0.02) { Actions {
                            changeVar("side", -1)
                            setEnable("gate4", true)
                            flash("fde047", 1.2)
                            delay(1.2) { Actions { setEnable("gate3", false) sound("bump") } }
                        } } }
                    }
                }
            }
        }
    }

    goal("pod", 1090, 250, 70, 100, "-- ESCAPE POD AWAY --")
}
