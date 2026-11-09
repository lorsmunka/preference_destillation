Window {
    size(8, 10)
    background("292524")
    joystick("hero", 70, 5)

    Entity { id("hero") startPosition(60, 80) color("60a5fa")
        Type { rectangle(46, 46) }
        Movement { speed(0) Control { movementControl(0) } } }

    Entity { id("key") startPosition(620, 120) color("facc15")
        Type { circle(24) }
        key("open", "hero", "pickup") }

    Entity { id("door") startPosition(0, 500) color("7c3aed")
        Type { rectangle(800, 30) }
        door("open", "clang") }

    goal("exit", 360, 880, 90, 60, "-- THROUGH THE DOOR --")

    Entity { id("bat") startPosition(100, 300) color("ef4444")
        Type { triangle(40, "up") }
        hostile("hero", "-- BITTEN --", "hurt")
        Movement { speed(3) Path { waypoint(60, 300) waypoint(700, 300) } } }
}
