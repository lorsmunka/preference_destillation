Window {
    size(8, 10)
    background("0f172a")
    music("bg_1")
    joystick("hero", 70, 5)
    winAt(1, "-- YOU MADE IT --", "win")

    Entity {
        id("hero")
        startPosition(60, 60)
        color("38bdf8")
        Type { rectangle(50, 50) }
        Movement { speed(0) Control { movementControl(0) } }
    }

    Entity {
        id("goal")
        startPosition(680, 880)
        color("facc15")
        Type { star(5, 40, 18) }
        Events {
            OnCollision("hero") { Actions { sound("pickup") burst(12, "facc15") updateBar(1) } }
        }
    }

    Entity {
        id("enemy")
        startPosition(300, 400)
        color("ef4444")
        Type { circle(30) }
        hostile("hero", "-- TOUCHED THE RED ONE --", "lose")
        Movement { speed(3) Path { waypoint(100, 450) waypoint(700, 450) } }
    }
}
