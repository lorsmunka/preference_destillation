# Tap balloons as they rise. Blue ones are 1 point, the gold one is 5, but it shrinks fast.
Window {
    size(8, 12)
    background("bfdbfe")
    music("bg_1")
    gravity(0)

    Bar("score") {
        initial(0)
        OnValue(30) { Actions { sound("win") gameOver("-- SKY CLEARED --") } }
    }
    Bar("escaped") {
        initial(0)
        OnValue(6) { Actions { sound("lose") gameOver("-- TOO MANY GOT AWAY --") } }
    }

    Entity {
        id("ceiling")
        startPosition(0, 1180)
        color("00000000")
        Type { rectangle(800, 20) }
    }

    Entity {
        id("balloon")
        template(true)
        startPosition(0, 0)
        color("3b82f6")
        Type { circle(34) }
        Movement { speed(1.8) AutoMove { direction(0, 1) } }
        Events {
            OnTapped {
                Actions {
                    sound("pop")
                    updateBar("score", 1)
                    burst(10, "3b82f6")
                    deleteEntity()
                }
            }
            OnCollision("ceiling") {
                Actions {
                    sound("blip")
                    updateBar("escaped", 1)
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("goldBalloon")
        template(true)
        startPosition(0, 0)
        color("f59e0b")
        Type { circle(40) }
        Movement { speed(2.6) AutoMove { direction(0, 1) } }
        Events {
            OnElapsedTime(1.5) { Actions { setScale(0.5) } }
            OnTapped {
                Actions {
                    sound("sparkle")
                    loopCount(5)
                    updateBar("score", 1)
                    burst(16, "fbbf24")
                    deleteEntity()
                }
            }
            OnCollision("ceiling") { Actions { deleteEntity() } }
        }
    }

    Wave(12, 1.2) { Actions { spawn("balloon", random(60, 700), 20) } }
    Wave(20, 0.8) {
        Actions {
            spawn("balloon", random(60, 700), 20)
            randomAction {
                Actions { spawn("goldBalloon", random(60, 700), 20) }
                Actions { spawn("balloon", random(60, 700), 20) }
                Actions { spawn("balloon", random(60, 700), 20) }
            }
        }
    }
}
