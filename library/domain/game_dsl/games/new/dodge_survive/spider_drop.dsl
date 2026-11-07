Window {
    size(7, 12)
    background("0f2a1d")
    music("bg_ambient")
    gravity(0)

    lives(3, "-- WRAPPED IN SILK --", "lose")

    Wave(15, 1.1) { Actions { spawn("spider", random(20, 630), 1120) } }
    Wave(15, 0.8) { Actions { spawn("spider", random(20, 630), 1120) } }
    Wave(20, 0.6) {
        Actions {
            spawn("spider", random(20, 630), 1120)
            spawn("dew", random(20, 670), 1150)
        }
    }
    Wave(3) {
        Actions {
            sound("win")
            burst("ant", 30, "fde68a")
            shake(4, 0.4)
            gameOver("-- HOME BEFORE DARK --")
        }
    }

    Entity { id("branch") startPosition(0, 1170) color("713f12")
        Type { rectangle(700, 30) } }

    Entity {
        id("ant")
        startPosition(330, 0)
        color("dc2626")
        Type { rectangle(40, 24) }
        Movement { speed(5) Control { movementControl(2) } }
    }

    Entity {
        id("spider")
        template(true)
        startPosition(0, 0)
        color("a855f7")
        Type { star(8, 24, 10) }
        Movement { speed(3.5) AutoMove { direction(0, -1) } }
        var("up", 0)
        Events {
            OnElapsedTime { Actions { updateSpeed(random(2.8, 4.4)) } }
            # climb back up at a random depth, or when the ground is reached
            OnElapsedTime(random(1.8, 3.4)) {
                Actions { whenVar("up", "==", 0) { Actions { changeVar("up", 1) setVelocity(0, 420) } } }
            }
            OnCollision {
                Actions {
                    whenVar("up", "==", 1) { Actions { deleteEntity() } }
                    whenVar("up", "==", 0) { Actions { changeVar("up", 1) setVelocity(0, 420) } }
                }
            }
            OnCollision("ant") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("ant", "ffffff", 0.3)
                    shake(5, 0.25)
                    burst(12, "a855f7")
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("dew")
        template(true)
        startPosition(0, 0)
        color("93c5fd")
        Type { circle(8) }
        Movement { speed(6) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("ant") { Actions {
                sound("bubble") updateBar("lives", -1) flash("ant", "93c5fd", 0.3) burst(8, "93c5fd") deleteEntity() } }
            OnCollision { Actions { burst(4, "bfdbfe") deleteEntity() } }
        }
    }
}
