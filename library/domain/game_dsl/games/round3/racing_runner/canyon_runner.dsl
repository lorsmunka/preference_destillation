Window {
    size(10, 6)
    background("7c2d12")
    music("bg_nature")
    gravity(900)

    lives(3, "-- TRIPPED --", "lose")
    Bar("meters") { initial(0) }
    Bar("heat") {
        initial(0)
        OnValue(8) { Actions { broadcast("hurry") } }
    }
    winAt("meters", 30, "-- OUT OF THE CANYON --", "win")

    Entity {
        id("runner")
        startPosition(120, 10)
        color("fde68a")
        Type { rectangle(50, 80) }
        Events {
            OnCollision("rock") {
                Actions {
                    sound("hurt")
                    shake(6, 0.3)
                    flash("ff0000", 0.25)
                    updateBar("lives", -1)
                }
            }
        }
    }

    Entity { id("jumpPad") startPosition(0, 0) color("00000000")
        gravity(0)
        Type { rectangle(1000, 600) }
        Events {
            OnTapped {
                Actions {
                    setVelocity("runner", 0, 560)
                    sound("jump")
                }
            }
        } }

    Entity {
        id("rock")
        template(true)
        startPosition(0, 0)
        color("a8a29e")
        gravity(0)
        Type { triangle(60, "up") }
        Movement { speed(4) AutoMove { direction(-1, 0) } }
        Events {
            OnCollision("runner") { Actions { deleteEntity() } }
            OnCollision {
                Actions {
                    updateBar("meters", 1)
                    deleteEntity()
                }
            }
            OnSignal("hurry") { Actions { updateSpeed(6.5) } }
        }
    }

    Entity {
        id("director")
        startPosition(500, 300)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.1) { Actions { ramp("heat", 0, 10, 30) } }
        }
    }

    Wave(15, 1.5) { Actions { spawn("rock", 900, 10) } }
    Wave(30, 1.1) { Actions { spawn("rock", 900, 10) } }
}
