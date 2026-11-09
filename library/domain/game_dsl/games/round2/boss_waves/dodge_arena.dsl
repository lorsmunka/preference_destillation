Window {
    size(8, 10)
    background("14110f")
    music("bg_6")
    lives(3, "-- OVERWHELMED --", "lose")
    winAt("time", 40, "-- YOU SURVIVED --", "win")

    Bar("time") { initial(0) }
    Bar("danger") {
        initial(1)
        OnValue(3) { Actions { broadcast("faster") } }
    }

    Wave(15, 1.2) { Actions { spawn("bolt", random(40, 740), 960) } }
    Wave(25, 0.6) { Actions { spawn("bolt", random(40, 740), 960) } }

    Entity {
        id("player")
        startPosition(380, 80)
        color("fde68a")
        Type { circle(18) }
        Movement { speed(6) Control { movementControl(0) } }
    }

    Entity {
        id("clock")
        startPosition(-50, -50)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(1) {
                repeatCount(-1)
                Actions { updateBar("time", 1) }
            }
            OnElapsedTime(15) { Actions { updateBar("danger", 2) } }
        }
    }

    HostilePatrol("sweeper", "player", "-- SWEPT AWAY --", 2.8, "f97316") {
        waypoint(40, 500)
        waypoint(700, 500)
        dwell(1.2)
    }

    Entity {
        id("sentry")
        startPosition(350, 900)
        color("dc2626")
        Type { rectangle(100, 40) }
        Events {
            OnSignal("faster") {
                Actions {
                    flash("ffffff", 0.5)
                    shake(5, 0.3)
                }
            }
            Sequence {
                Phase(5, 2) {
                    Actions {
                        setColor("ef4444")
                        delay(0.5) { Actions { spawnHere("bolt") setColor("dc2626") } }
                    }
                }
                Phase(-1, 1) {
                    Actions {
                        setColor("ef4444")
                        delay(0.4) { Actions { spawnHere("bolt") setColor("dc2626") } }
                    }
                }
            }
        }
    }

    Entity {
        id("bolt")
        template(true)
        startPosition(0, 0)
        color("fb923c")
        Type { triangle(28, "down") }
        Movement { speed(3.6) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("player") {
                Actions {
                    updateBar("lives", -1)
                    sound("hurt")
                    flash("player", "ff0000", 0.3)
                    burst(10, "fb923c")
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }
}
