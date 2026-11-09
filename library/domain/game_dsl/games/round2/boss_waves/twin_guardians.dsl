Window {
    size(12, 8)
    background("1a1033")
    music("bg_5")
    lives(3, "-- THE GUARDIANS WIN --", "lose")

    Bar("charge") {
        initial(0)
        OnValue(4) { Actions { broadcast("zap") } }
    }
    Bar("boss") {
        initial(3)
        OnValue(0) { Actions { sound("win") gameOver("-- GUARDIAN CORE SHATTERED --") } }
    }

    Wave(5, 5) {
        Actions { spawn("crystal", random(100, 1000), random(100, 650)) }
    }

    HostilePatrol("guardA", "player", "-- CRUSHED --", 3.2, "f43f5e") {
        waypoint(100, 650)
        waypoint(1000, 650)
        dwell(0.5)
    }
    HostilePatrol("guardB", "player", "-- CRUSHED --", 2.4, "fb923c") {
        waypoint(1000, 100)
        waypoint(600, 400)
        waypoint(100, 100)
    }

    Entity {
        id("player")
        startPosition(560, 360)
        color("67e8f9")
        Type { circle(22) }
        Movement { speed(6) Control { movementControl(0) } }
    }

    Entity {
        id("core")
        startPosition(530, 330)
        color("c084fc")
        Type { star(5, 50, 22) }
        solid(true)
        Events {
            OnSignal("zap") {
                Actions {
                    sound("laser")
                    updateBar("boss", -1)
                    updateBar("charge", -4)
                    flash("ffffff", 0.4)
                    shake(7, 0.4)
                    burst(16, "c084fc")
                }
            }
            Sequence {
                Phase(3, 3) { Actions { spawnHere("spark") } }
                Phase(-1, 1.5) { Actions { spawnHere("spark") } }
            }
        }
    }

    Entity {
        id("crystal")
        template(true)
        startPosition(0, 0)
        color("5eead4")
        Type { diamond(34, 44) }
        Events {
            OnCollision("player") {
                Actions {
                    updateBar("charge", 1)
                    sound("pickup")
                    burst(6, "5eead4")
                    deleteEntity()
                }
            }
            OnElapsedTime(7) { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("spark")
        template(true)
        startPosition(0, 0)
        color("fde047")
        Type { circle(12) }
        Movement { speed(2.2) chase("player") }
        Events {
            OnCollision("player") {
                Actions {
                    updateBar("lives", -1)
                    flash("player", "ff0000", 0.3)
                    sound("hurt")
                    burst(8, "fde047")
                    deleteEntity()
                }
            }
            OnElapsedTime(6) { Actions { burst(5, "fde047") deleteEntity() } }
        }
    }
}
