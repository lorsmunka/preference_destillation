Window {
    size(8, 12)
    background("0c4a6e")
    music("bg_nature")
    gravity(0)
    lives(3, "-- DROPPED TOO MANY --", "lose")
    winAt(12, "-- RICH! --", "win2")

    Entity {
        id("basket")
        startPosition(340, 60)
        color("f59e0b")
        Type { rectangle(120, 30) }
        Movement { speed(6) Control { movementControl(2) } }
    }

    Entity {
        id("coin")
        template(true)
        startPosition(0, 0)
        color("fde047")
        Type { circle(22) }
        Movement { speed(3) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("basket") { Actions { sound("coin") updateBar(1) deleteEntity() } }
            OnCollision { Actions { updateBar("lives", -1) deleteEntity() } }
        }
    }

    Entity {
        id("sky")
        startPosition(400, 1150)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(1.2) { repeatCount(-1) Actions { spawn("coin", random(30, 740), 1150) } }
        }
    }
}
