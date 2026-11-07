Window {
    size(8, 12)
    background("86efac")
    music("bg_nature")
    gravity(0)

    lives(3, "-- THE DUCKLING RAN HOME --", "lose")
    Bar("gulps") {
        initial(0)
        OnValue(15) {
            Actions {
                sound("sparkle")
                broadcast("bigFrog")
            }
        }
    }

    Entity {
        id("pond")
        startPosition(280, 1090)
        color("38bdf8")
        Type { rectangle(240, 90) }
    }

    Entity {
        id("duckling")
        startPosition(380, 60)
        color("fde047")
        Type { circle(20) }
        Movement {
            speed(0.45)
            Path {
                waypoint(380, 60)
                waypoint(120, 300)
                waypoint(600, 520)
                waypoint(180, 760)
                waypoint(560, 950)
                waypoint(380, 1080)
                dwell(0.5)
            }
        }
        Events {
            OnCollision("pond") {
                Actions {
                    sound("win")
                    shake(4, 0.4)
                    burst(30, "38bdf8")
                    gameOver("-- SAFE IN THE POND --")
                }
            }
            OnCollision("mosquito") {
                Actions {
                    deleteCollider()
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("ef4444", 0.3)
                    shake(4, 0.2)
                }
            }
            OnCollision("beetle") {
                Actions {
                    deleteCollider()
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("ef4444", 0.3)
                    shake(6, 0.3)
                }
            }
        }
    }

    Entity {
        id("frog")
        startPosition(372, 160)
        color("15803d")
        Type { rectangle(56, 40) }
        Movement {
            speed(7)
            Control { movementControl(0) }
        }
        Events {
            OnElapsedTime(0.5) {
                repeatCount(-1)
                Actions { spawnHere("spit") }
            }
            OnCollision("mosquito") {
                Actions {
                    deleteCollider()
                    sound("pop")
                    burst(6, "a3e635")
                    updateBar("gulps", 1)
                }
            }
            OnSignal("bigFrog") {
                Actions {
                    setScale(1.4)
                    flash("fde047", 0.5)
                }
            }
        }
    }

    Entity { id("spit") template(true) color("bef264") Type { circle(8) }
        Movement { speed(7) AutoMove { direction(0, 1) } }
        Events { OnCollision { Actions { deleteEntity() } } } }

    Entity { id("mosquito") template(true) color("44403c") Type { circle(10) }
        Movement { speed(1.5) chase("duckling") }
        Events { OnCollision("spit") { Actions { deleteCollider() sound("pop") burst(6, "78716c") updateBar("gulps", 1) deleteEntity() } } } }

    Entity { id("beetle") template(true) color("1e3a8a") Type { polygon(6, 24) }
        var("hp", 3)
        Movement { speed(0.6) chase("duckling") }
        Events { OnCollision("spit") { Actions {
            deleteCollider() changeVar("hp", -1) flash("ffffff", 0.1) sound("clang")
            whenVar("hp", "<=", 0) { Actions { sound("pop") burst(14, "3b82f6") updateBar("gulps", 1) deleteEntity() } } } } } }

    Entity { id("reedL") startPosition(0, 640) color("4d7c0f") Type { rectangle(30, 160) }
        Events {
            OnElapsedTime(3) { repeatCount(14) Actions { spawnHere("mosquito") } }
            OnElapsedTime(9) { repeatCount(4) Actions { spawnHere("beetle") } }
        } }
    Entity { id("reedR") startPosition(770, 820) color("4d7c0f") Type { rectangle(30, 160) }
        Events {
            OnElapsedTime(2.4) { repeatCount(17) Actions { spawnHere("mosquito") } }
            OnElapsedTime(11) { repeatCount(3) Actions { spawnHere("beetle") } }
        } }
    Entity { id("reedTop") startPosition(40, 1100) color("4d7c0f") Type { rectangle(160, 30) }
        Events {
            OnElapsedTime(random(3, 5)) { repeatCount(10) Actions { spawnHere("mosquito") spawnHere("mosquito") } }
        } }
}
