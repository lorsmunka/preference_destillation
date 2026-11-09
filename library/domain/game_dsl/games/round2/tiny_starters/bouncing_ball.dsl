Window {
    size(8, 8)
    background("0a0e27")
    music("bg_2")
    winAt(5, "-- CAUGHT IT 5 TIMES --", "win")

    Entity {
        id("ball")
        startPosition(100, 200)
        color("4ade80")
        Type { circle(40) }
        Movement { speed(4) AutoMove { direction(1, 1) } }
        Events {
            OnCollision { Actions { sound("bump") bounceBack() } }
            OnCollision("bumper") { Actions { sound("thud") bounceBack() } }
            OnTapped("ball") {
                Actions {
                    sound("blip")
                    updateBar(1)
                    updateSpeed(5.5)
                    burst(8, "86efac")
                }
            }
        }
    }

    Entity {
        id("bumper")
        startPosition(330, 330)
        color("a78bfa")
        Type { rectangle(140, 140) }
        Events { OnCollision("ball") { Actions { flash("ffffff", 0.15) } } }
    }

    Entity {
        id("timer")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(25.0) { Actions { sound("lose") gameOver("-- THE BALL GOT AWAY --") } }
        }
    }
}
