Window {
    size(7, 12)
    background("0c4a6e")
    music("bg_ambient")
    gravity(0)

    lives(3, "-- CAPSIZED --", "lose")
    Bar("gates") {
        initial(0)
        OnValue(6) { Actions {
            sound("win")
            gameOver("-- FIRST DOWN THE RIVER --")
        } }
    }
    Bar("current") {
        initial(1)
        OnValue(5) { Actions { broadcast("rapids") } }
    }

    Entity { id("bankLeft") startPosition(0, 0) color("166534")
        Type { rectangle(60, 1200) } solid(true) }
    Entity { id("bankRight") startPosition(640, 0) color("166534")
        Type { rectangle(60, 1200) } solid(true) }

    Entity {
        id("boat")
        startPosition(330, 120)
        color("fde047")
        Type { triangle(50, "up") }
        Movement { speed(5) Control { movementControl(0) } }
        Events {
            OnCollision("log") {
                Actions {
                    sound("thud")
                    shake(6, 0.3)
                    flash("ff0000", 0.25)
                    updateBar("lives", -1)
                    deleteCollider()
                }
            }
            OnCollision("buoy") {
                Actions {
                    sound("bell")
                    updateBar("gates", 1)
                    burst(8, "f97316")
                    deleteCollider()
                }
            }
        }
    }

    Entity {
        id("log")
        template(true)
        startPosition(0, 0)
        color("92400e")
        Type { rectangle(120, 36) }
        Movement { speed(2.5) AutoMove { direction(0, -1) } }
        Events {
            OnCollision { Actions { deleteEntity() } }
            OnSignal("rapids") { Actions { updateSpeed(4) } }
        }
    }
    Entity {
        id("buoy")
        template(true)
        startPosition(0, 0)
        color("f97316")
        Type { circle(26) }
        Movement { speed(2.5) AutoMove { direction(0, -1) } }
        Events {
            OnCollision { Actions { deleteEntity() } }
            OnSignal("rapids") { Actions { updateSpeed(4) } }
        }
    }

    Entity {
        id("river")
        startPosition(350, 20)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(1.3) {
                repeatCount(-1)
                Actions { spawn("log", random(70, 500), 1100) }
            }
            OnElapsedTime(3) {
                repeatCount(-1)
                Actions { spawn("buoy", random(90, 560), 1100) }
            }
            OnElapsedTime(2) {
                Actions { ramp("current", 1, 6, 20) }
            }
        }
    }
}
