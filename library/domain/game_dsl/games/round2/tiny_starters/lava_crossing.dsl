Window {
    size(8, 10)
    background("1c1917")
    joystick("hero", 70, 5)
    winAt(1, "-- CROSSED THE LAVA --", "win")

    Entity {
        id("hero")
        startPosition(380, 30)
        color("4ade80")
        Type { circle(24) }
        Movement { speed(0) Control { movementControl(0) } }
    }
    Entity {
        id("far_bank")
        startPosition(0, 930)
        color("16a34a")
        Type { rectangle(800, 60) }
        Events { OnCollision("hero") { Actions { sound("win") updateBar(1) } } }
    }
    Entity {
        id("lava1")
        startPosition(0, 350)
        color("ea580c")
        Type { rectangle(160, 50) }
        hostile("hero", "-- BURNED --", "death")
        Movement { speed(3) AutoMove { direction(1, 0) } }
        Events { OnCollision { Actions { bounceBack() } } }
    }
    Entity {
        id("lava2")
        startPosition(600, 600)
        color("dc2626")
        Type { rectangle(200, 50) }
        hostile("hero", "-- BURNED --", "death")
        Movement { speed(4) AutoMove { direction(-1, 0) } }
        Events { OnCollision { Actions { bounceBack() } } }
    }
}
