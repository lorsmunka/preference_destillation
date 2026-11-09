Window {
    size(7, 11)
    background("111827")
    gravity(0)
    Bar("dodged") { initial(0) OnValue(10) { Actions { sound("win") gameOver("-- YOU DODGED THEM ALL --") } } }

    Entity {
        id("player")
        startPosition(310, 50)
        color("34d399")
        Type { rectangle(80, 40) }
        Movement { speed(6) Control { movementControl(2) } }
    }

    Entity {
        id("block")
        template(true)
        startPosition(0, 0)
        color("fb923c")
        Type { rectangle(70, 70) }
        Movement { speed(3.5) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("player") { Actions { sound("hurt") gameOver("-- SQUASHED --") } }
            OnCollision { Actions { updateBar("dodged", 1) deleteEntity() } }
        }
    }

    Entity {
        id("dropper")
        startPosition(350, 1050)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(1.0) { repeatCount(-1) Actions { spawn("block", random(10, 620), 1040) } }
        }
    }
}
