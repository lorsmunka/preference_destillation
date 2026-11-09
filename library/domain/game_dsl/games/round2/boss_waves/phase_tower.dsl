# A three-phase boss: its pattern changes as its health bar drops.
Window {
    size(8, 14)
    background("07131f")
    music("bg_4")
    lives(5, "-- TOWER STANDS --", "lose")

    Bar("health") {
        initial(30)
        OnValue(20) { Actions { broadcast("phase2") } }
        OnValue(10) { Actions { broadcast("phase3") } }
        OnValue(0) { Actions { sound("win") gameOver("-- TOWER FALLEN --") } }
    }

    Entity {
        id("player")
        startPosition(380, 100)
        color("4ade80")
        Type { circle(26) }
        Movement { speed(6.5) Control { movementControl(0) } }
        Events {
            OnElapsedTime(0.28) {
                repeatCount(-1)
                Actions { spawnHere("bolt") }
            }
        }
    }

    Entity {
        id("boss")
        startPosition(300, 1160)
        color("38bdf8")
        Type { rectangle(200, 90) }
        Movement { speed(1.8) Path { waypoint(60, 1160) waypoint(540, 1160) } }
        Events {
            OnCollision("bolt") {
                Actions {
                    updateBar("health", -1)
                    flash("ffffff", 0.08)
                }
            }
            Sequence {
                Phase(-1, 1.6) {
                    Actions {
                        spawnHere("slug")
                    }
                }
            }
            OnSignal("phase2") {
                Actions {
                    setColor("facc15")
                    shake(5, 0.4)
                    sound("buzz")
                    spawn("fan", 100, 1000)
                    spawn("fan", 600, 1000)
                }
            }
            OnSignal("phase3") {
                Actions {
                    setColor("ef4444")
                    shake(8, 0.6)
                    sound("buzz")
                    spawn("seeker", 400, 1000)
                }
            }
            OnElapsedTime(1.1) {
                repeatCount(-1)
                Actions {
                    whenBar("health", "<=", 20) {
                        Actions { spawnHere("slug") }
                    }
                }
            }
        }
    }

    Entity {
        id("bolt")
        template(true)
        startPosition(0, 0)
        color("a7f3d0")
        Type { rectangle(8, 24) }
        Movement { speed(10) AutoMove { direction(0, 1) } }
        Events {
            OnCollision("boss") { Actions { deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("slug")
        template(true)
        startPosition(0, 0)
        color("f472b6")
        Type { circle(18) }
        Movement { speed(3.2) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("player") {
                Actions {
                    updateBar("lives", -1)
                    sound("hurt")
                    burst(8, "f472b6")
                    flash("player", "ff0000", 0.2)
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("fan")
        template(true)
        startPosition(0, 0)
        color("fbbf24")
        Type { triangle(40, "down") }
        Movement { speed(2) AutoMove { direction(0, -1) } }
        Events {
            OnElapsedTime(0.9) {
                repeatCount(-1)
                Actions { spawnHere("slug") }
            }
            OnCollision("bolt") {
                Actions { burst(8, "fbbf24") sound("pop") deleteEntity() }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("seeker")
        template(true)
        startPosition(0, 0)
        color("ef4444")
        Type { polygon(5, 30) }
        Movement { speed(1.5) chase("player") }
        Events {
            OnCollision("player") {
                Actions {
                    updateBar("lives", -2)
                    shake(6, 0.3)
                    burst(14, "ef4444")
                    deleteEntity()
                }
            }
            OnCollision("bolt") {
                Actions { burst(8, "ef4444") deleteEntity() }
            }
        }
    }
}
