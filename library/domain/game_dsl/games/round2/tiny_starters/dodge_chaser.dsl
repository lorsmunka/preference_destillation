Window {
    size(8, 10)
    background("14532d")
    joystick("runner", 70, 5.5)
    Bar("seconds") { initial(0) OnValue(20) { Actions { sound("win") gameOver("-- SURVIVED 20 SECONDS --") } } }

    Entity {
        id("runner")
        startPosition(380, 480)
        color("e0f2fe")
        Type { circle(25) }
        Movement { speed(0) Control { movementControl(0) } }
    }

    Entity {
        id("wolf")
        startPosition(60, 840)
        color("57534e")
        Type { polygon(6, 32) }
        hostile("runner", "-- THE WOLF GOT YOU --", "death")
        Movement { speed(2.6) chase("runner") }
    }

    Entity {
        id("clock")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(1.0) { repeatCount(-1) Actions { updateBar("seconds", 1) } }
        }
    }
}
