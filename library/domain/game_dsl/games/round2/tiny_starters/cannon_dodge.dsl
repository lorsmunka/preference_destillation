Window {
    size(8, 10)
    background("1e293b")
    gravity(0)
    Bar("score") { initial(0) OnValue(6) { Actions { sound("win") gameOver("-- ALL SIX COINS --") } } }

    Entity {
        id("hero")
        startPosition(60, 460)
        color("2dd4bf")
        Type { rectangle(50, 50) }
        Movement { speed(6) Control { movementControl(1) } }
    }

    Entity {
        id("shot")
        template(true)
        startPosition(0, 0)
        color("f97316")
        Type { circle(20) }
        Movement { speed(5) AutoMove { direction(-1, 0) } }
        Events {
            OnCollision("hero") { Actions { sound("hurt") gameOver("-- HIT BY A CANNONBALL --") } }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("cannon")
        startPosition(700, 0)
        color("78716c")
        Type { rectangle(60, 60) }
        Events {
            OnElapsedTime(0.9) {
                repeatCount(-1)
                Actions { spawn("shot", 700, random(30, 900)) }
            }
        }
    }

    Entity {
        id("coin")
        startPosition(100, 700)
        color("fde047")
        Type { circle(20) }
        Events {
            OnCollision("hero") {
                Actions { sound("coin") updateBar("score", 1) setPosition("coin", random(60, 300), random(40, 900), 0) }
            }
        }
    }
}
