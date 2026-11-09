Window {
    size(8, 10)
    background("121826")
    music("bg_3")
    winAt(10, "-- 10 BOUNCES --", "win")

    Entity {
        id("paddle")
        startPosition(340, 60)
        color("38bdf8")
        Type { rectangle(140, 20) }
        Movement { speed(6) Control { movementControl(2) } }
    }

    Entity {
        id("floor")
        startPosition(0, 0)
        color("7f1d1d")
        Type { rectangle(800, 16) }
    }

    Entity {
        id("ball")
        startPosition(400, 500)
        color("fbbf24")
        Type { circle(18) }
        Movement { speed(4) AutoMove { direction(1, -1) } }
        Events {
            OnCollision("paddle") { Actions { sound("blip") updateBar(1) bounceBack() } }
            OnCollision("floor") { Actions { sound("death") gameOver("-- DROPPED THE BALL --") } }
            OnCollision { Actions { bounceBack() } }
        }
    }
}
