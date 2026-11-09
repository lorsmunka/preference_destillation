Window {
    size(8, 12)
    background("0b1020")
    music("bg_3")
    lives(3, "-- SHIP LOST --", "lose")
    winAt("boss", 0, "-- CORE DESTROYED --", "win")

    Bar("boss") {
        initial(15)
        OnValue(8) { Actions { broadcast("phase2") } }
    }

    Entity {
        id("player")
        startPosition(360, 80)
        color("22d3ee")
        Type { triangle(60, "up") }
        Movement { speed(6.5) Control { movementControl(2) } }
        Events {
            OnElapsedTime(0.35) {
                repeatCount(-1)
                Actions { spawnHere("bullet") }
            }
        }
    }

    Entity {
        id("boss")
        startPosition(300, 1000)
        color("a855f7")
        Type { polygon(6, 100) }
        Movement { speed(2) Path { waypoint(100, 1000) waypoint(500, 1000) } }
        Events {
            OnCollision("bullet") {
                Actions {
                    updateBar("boss", -1)
                    flash("ffffff", 0.1)
                    burst(6, "a855f7")
                }
            }
            Sequence {
                Phase(4, 2.2) {
                    Actions {
                        spawnHere("orb")
                    }
                }
                Phase(-1, 1.4) {
                    Actions {
                        spawnHere("orb")
                        spawn("orb", 100, 900)
                    }
                }
            }
            OnSignal("phase2") {
                Actions {
                    shake(6, 0.4)
                    flash("ef4444", 0.6)
                    setColor("ef4444")
                }
            }
        }
    }

    Entity {
        id("bullet")
        template(true)
        startPosition(0, 0)
        color("fde047")
        Type { rectangle(8, 26) }
        Movement { speed(9) AutoMove { direction(0, 1) } }
        Events {
            OnCollision("boss") { Actions { deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("orb")
        template(true)
        startPosition(0, 0)
        color("fb7185")
        Type { circle(20) }
        Movement { speed(3.5) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("player") {
                Actions {
                    updateBar("lives", -1)
                    sound("hurt")
                    shake(5, 0.25)
                    burst(10, "fb7185")
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }
}
