# Catch falling coins in the cart; miss three and the shift is over.
Window {
    size(8, 12)
    background("1b2a41")
    music("bg_2")
    gravity(0)

    Bar("score") { initial(0) }
    winAt("score", 25, "-- PAYDAY! --", "win")
    lives(3, "-- THE TILL IS EMPTY --", "lose")

    Entity {
        id("player")
        startPosition(340, 60)
        color("38bdf8")
        Type { rectangle(120, 30) }
        Movement { speed(6) Control { movementControl(2) } }
    }

    Entity {
        id("coin")
        template(true)
        startPosition(0, 0)
        color("facc15")
        Type { circle(20) }
        Movement { speed(2.5) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("player") {
                Actions {
                    sound("coin")
                    updateBar("score", 1)
                    burst(8, "facc15")
                    deleteEntity()
                }
            }
            OnCollision {
                Actions {
                    sound("buzz")
                    updateBar("lives", -1)
                    deleteEntity()
                }
            }
        }
    }

    Wave(10, 1.4) { Actions { spawn("coin", random(40, 720), 1160) } }
    Wave(10, 1.0) { Actions { spawn("coin", random(40, 720), 1160) } }
    Wave(30, 0.7) { Actions { spawn("coin", random(40, 720), 1160) } }
}
