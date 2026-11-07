Window {
    size(8, 10)
    background("3b0764")
    music("bg_3")
    gravity(450)

    lives(3, "-- THE CROWD BOOS --", "lose")
    Bar("score") {
        initial(0)
        OnValue(10) {
            Actions {
                sound("sparkle")
                spawn("ballB", 200, 820)
            }
        }
        OnValue(22) {
            Actions {
                sound("sparkle")
                spawn("ballB", 580, 820)
            }
        }
    }
    winAt("score", 35, "-- STANDING OVATION --", "win")

    wallBox(0, 0, 800, 1000, "facc15", 16)
    paddle("clown", 335, 60, 130, 22)

    Entity {
        id("ballA")
        startPosition(390, 760)
        color("f43f5e")
        Type { circle(18) }
        Movement {
            speed(1.5)
            AutoMove { direction(0.6, -1) }
        }
        Events {
            OnCollision("clown") {
                Actions {
                    sound("bump")
                    bounceBack()
                    updateBar("score", 1)
                    burst(8, "f43f5e")
                }
            }
            OnCollision("wallL") { Actions { sound("click") bounceBack(0) } }
            OnCollision("wallR") { Actions { sound("click") bounceBack(0) } }
            OnCollision("acrobat") { Actions { sound("pop") bounceBack(0) } }
            OnCollision("hoop") {
                Actions {
                    sound("bell")
                    updateBar("score", 3)
                    burst(14, "fde047")
                }
            }
            OnCollision("wallB") {
                Actions {
                    sound("hurt")
                    shake(7, 0.4)
                    burst(20, "f43f5e")
                    updateBar("lives", -1)
                    setPosition("ballA", 390, 760, 0)
                    setVelocity(random(-120, 120), 0)
                }
            }
        }
    }

    # bonus balls join the act at 10 and 22; a dropped bonus ball is gone
    Entity {
        id("ballB")
        template(true)
        startPosition(0, 0)
        color("38bdf8")
        Type { circle(18) }
        Movement {
            speed(1)
            AutoMove { direction(0.4, -1) }
        }
        Events {
            OnCollision("clown") {
                Actions {
                    sound("bump")
                    bounceBack()
                    updateBar("score", 1)
                    burst(8, "38bdf8")
                }
            }
            OnCollision("wallL") { Actions { sound("click") bounceBack(0) } }
            OnCollision("wallR") { Actions { sound("click") bounceBack(0) } }
            OnCollision("acrobat") { Actions { sound("pop") bounceBack(0) } }
            OnCollision("hoop") {
                Actions {
                    sound("bell")
                    updateBar("score", 3)
                    burst(14, "fde047")
                }
            }
            OnCollision("wallB") {
                Actions {
                    sound("hurt")
                    shake(7, 0.4)
                    burst(20, "38bdf8")
                    updateBar("lives", -1)
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("hoop")
        startPosition(120, 600)
        color("fde047")
        Type { ring(40, 32) }
        Movement {
            speed(1.4)
            Path {
                waypoint(120, 600)
                waypoint(600, 600)
            }
        }
        Events {
            OnElapsedTime(1.5) {
                repeatCount(-1)
                Actions { burst(4, "fde047") }
            }
        }
    }

    Entity {
        id("acrobat")
        startPosition(150, 420)
        color("f9a8d4")
        Type { rectangle(36, 60) }
        Movement {
            speed(1)
            Path {
                waypoint(150, 420)
                waypoint(610, 420)
                dwell(0.8)
            }
        }
    }
}
