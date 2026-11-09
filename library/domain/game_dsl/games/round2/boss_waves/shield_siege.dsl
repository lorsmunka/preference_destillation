# Survive three waves of raiders while two shields soak up their fire.
Window {
    size(9, 12)
    background("101827")
    music("bg_1")
    lives(4, "-- BASE FALLEN --", "lose")

    Bar("kills") {
        initial(0)
        OnValue(24) { Actions { sound("win") gameOver("-- ALL WAVES CLEARED --") } }
    }

    Wave(8, 1.2) {
        Actions { spawn("raider", random(60, 780), 1100) }
    }
    Wave(8, 0.9) {
        Actions { spawn("raider", random(60, 780), 1100) }
    }
    Wave(8, 0.7) {
        Actions { spawn("raider", random(60, 780), 1100) }
    }

    Entity {
        id("player")
        startPosition(420, 60)
        color("38bdf8")
        Type { rectangle(70, 28) }
        Movement { speed(6) Control { movementControl(2) } }
        Events {
            OnElapsedTime(0.5) {
                repeatCount(-1)
                Actions { spawnHere("bullet") }
            }
        }
    }

    Entity {
        id("shieldLeft")
        startPosition(180, 220)
        color("34d399")
        Type { rectangle(160, 20) }
        var("hp", 4)
        Events {
            OnCollision("shot") {
                Actions {
                    changeVar("hp", -1)
                    flash("ffffff", 0.1)
                    whenVar("hp", "<=", 0) {
                        Actions { burst(12, "34d399") deleteEntity() }
                    }
                }
            }
        }
    }

    Entity {
        id("shieldRight")
        startPosition(560, 220)
        color("34d399")
        Type { rectangle(160, 20) }
        var("hp", 4)
        Events {
            OnCollision("shot") {
                Actions {
                    changeVar("hp", -1)
                    flash("ffffff", 0.1)
                    whenVar("hp", "<=", 0) {
                        Actions { burst(12, "34d399") deleteEntity() }
                    }
                }
            }
        }
    }

    Entity {
        id("raider")
        template(true)
        startPosition(0, 0)
        color("f87171")
        Type { diamond(50, 50) }
        Movement { speed(1.6) AutoMove { direction(0, -1) } }
        Events {
            OnElapsedTime(0.8) {
                repeatCount(-1)
                Actions { spawnHere("shot") }
            }
            OnCollision("bullet") {
                Actions {
                    updateBar("kills", 1)
                    sound("pop")
                    burst(8, "f87171")
                    deleteEntity()
                }
            }
            OnCollision("player") {
                Actions {
                    updateBar("lives", -1)
                    shake(6, 0.3)
                    deleteEntity()
                }
            }
            OnCollision {
                Actions { updateBar("lives", -1) sound("hurt") deleteEntity() }
            }
        }
    }

    Entity {
        id("bullet")
        template(true)
        startPosition(0, 0)
        color("fef08a")
        Type { rectangle(8, 24) }
        Movement { speed(9) AutoMove { direction(0, 1) } }
        Events {
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("shot")
        template(true)
        startPosition(0, 0)
        color("fb923c")
        Type { circle(10) }
        Movement { speed(3.5) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("player") {
                Actions {
                    updateBar("lives", -1)
                    flash("player", "ff0000", 0.25)
                    sound("hurt")
                    deleteEntity()
                }
            }
            OnCollision("shieldLeft") { Actions { deleteEntity() } }
            OnCollision("shieldRight") { Actions { deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }
}
