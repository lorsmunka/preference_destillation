Window {
    size(8, 12)
    background("0f172a")
    music("bg_ambient")
    lives(3, "-- OUTMATCHED --", "lose")

    Bar("nodes") {
        initial(2)
        OnValue(0) { Actions { broadcast("exposed") } }
    }
    Bar("boss") {
        initial(8)
        OnValue(0) { Actions { sound("win") gameOver("-- BEARER DEFEATED --") } }
    }

    Entity {
        id("player")
        startPosition(370, 70)
        color("22d3ee")
        Type { circle(28) }
        Movement { speed(6) Control { movementControl(2) } }
        Events {
            OnElapsedTime(0.4) {
                repeatCount(-1)
                Actions { spawnHere("bullet") }
            }
        }
    }

    Entity {
        id("boss")
        startPosition(310, 960)
        color("f59e0b")
        Type { circle(90) }
        Events {
            OnCollision("bullet") {
                Actions {
                    sound("clang")
                    burst(4, "f59e0b")
                    whenBar("nodes", "<=", 0) {
                        Actions {
                            updateBar("boss", -1)
                            flash("ffffff", 0.2)
                            shake(5, 0.3)
                        }
                    }
                }
            }
            OnSignal("exposed") {
                Actions { setColor("fb7185") sound("buzz") }
            }
            Sequence {
                Phase(-1, 2) { Actions { spawn("fireball", 330, 940) spawn("fireball", 470, 940) } }
            }
        }
    }

    Entity {
        id("shield")
        startPosition(250, 860)
        color("60a5fa")
        Type { rectangle(300, 24) }
        solid(true)
        Events {
            OnSignal("exposed") {
                Actions { burst(20, "60a5fa") sound("glass") deleteEntity() }
            }
            OnCollision("bullet") { Actions { flash("ffffff", 0.05) } }
        }
    }

    Entity {
        id("nodeLeft")
        startPosition(60, 700)
        color("4ade80")
        Type { star(5, 40, 18) }
        var("hp", 6)
        Movement { speed(1.5) Path { waypoint(60, 700) waypoint(60, 520) } }
        Events {
            OnCollision("bullet") {
                Actions {
                    changeVar("hp", -1)
                    flash("ffffff", 0.08)
                    whenVar("hp", "<=", 0) {
                        Actions { updateBar("nodes", -1) burst(14, "4ade80") deleteEntity() }
                    }
                }
            }
        }
    }

    Entity {
        id("nodeRight")
        startPosition(620, 700)
        color("4ade80")
        Type { star(5, 40, 18) }
        var("hp", 6)
        Movement { speed(1.5) Path { waypoint(620, 520) waypoint(620, 700) } }
        Events {
            OnCollision("bullet") {
                Actions {
                    changeVar("hp", -1)
                    flash("ffffff", 0.08)
                    whenVar("hp", "<=", 0) {
                        Actions { updateBar("nodes", -1) burst(14, "4ade80") deleteEntity() }
                    }
                }
            }
        }
    }

    Entity {
        id("bullet")
        template(true)
        startPosition(0, 0)
        color("a5f3fc")
        Type { rectangle(8, 22) }
        Movement { speed(9) AutoMove { direction(0, 1) } }
        Events {
            OnCollision("shield") { Actions { deleteEntity() } }
            OnCollision("boss") { Actions { deleteEntity() } }
            OnCollision("nodeLeft") { Actions { deleteEntity() } }
            OnCollision("nodeRight") { Actions { deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("fireball")
        template(true)
        startPosition(0, 0)
        color("ef4444")
        Type { circle(22) }
        Movement { speed(3.3) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("player") {
                Actions {
                    updateBar("lives", -1)
                    sound("hurt")
                    flash("player", "ff0000", 0.25)
                    shake(6, 0.3)
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }
}
