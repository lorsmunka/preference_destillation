# Reef Diver: swim across wide reefs with the joystick. Pearls are the bonus, the treasure chest the goal.
Window {
    size(8, 9)
    worldSize(24, 9)
    cameraFollow("hero")
    background("0c4a6e")
    music("bg_ambient")
    lives(3, "-- OUT OF BREATH --", "lose")
    Bar("pearls") { initial(0) }

    Level {
        joystick("hero", 80, 5)
        wallBox(0, 0, 2400, 900, "082f49", 14)
        Entity {
            id("hero")
            startPosition(60, 420)
            color("fbbf24")
            Type { circle(26) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnCollision("pearl1") { Actions { sound("bubble") updateBar("pearls", 1) deleteCollider() } }
                OnCollision("pearl2") { Actions { sound("bubble") updateBar("pearls", 1) deleteCollider() } }
                OnCollision("jelly") { Actions { sound("hurt") updateBar("lives", -1) setPosition("hero", 60, 420, 0) } }
                OnCollision("gate") { Actions { sound("win2") burst(16, "22d3ee") nextLevel() } }
            }
        }
        Entity { id("pearl1") startPosition(900, 600) color("f8fafc") Type { circle(14) } }
        Entity { id("pearl2") startPosition(1700, 200) color("f8fafc") Type { circle(14) } }
        Entity { id("gate") startPosition(2250, 380) color("22d3ee") Type { ring(50, 28) } }
        Entity { id("coral1") startPosition(500, 14) color("fb7185") solid(true) Type { rectangle(120, 400) } }
        Entity { id("coral2") startPosition(1300, 486) color("f472b6") solid(true) Type { rectangle(120, 400) } }
        Entity { id("jelly") startPosition(1000, 300) color("c084fc") Type { circle(30) }
            Movement { speed(2.5) Path { waypoint(1000, 120) waypoint(1000, 780) } } }
    }

    Level {
        joystick("hero", 80, 5)
        wallBox(0, 0, 2400, 900, "172554", 14)
        Entity {
            id("hero")
            startPosition(60, 420)
            color("fbbf24")
            Type { circle(26) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnCollision("pearl1") { Actions { sound("bubble") updateBar("pearls", 1) deleteCollider() } }
                OnCollision("pearl2") { Actions { sound("bubble") updateBar("pearls", 1) deleteCollider() } }
                OnCollision("eel") { Actions { sound("hurt") updateBar("lives", -1) setPosition("hero", 60, 420, 0) } }
                OnCollision("gate") { Actions { sound("win2") burst(16, "22d3ee") nextLevel() } }
            }
        }
        Entity { id("pearl1") startPosition(700, 700) color("f8fafc") Type { circle(14) } }
        Entity { id("pearl2") startPosition(1900, 160) color("f8fafc") Type { circle(14) } }
        Entity { id("gate") startPosition(2250, 380) color("22d3ee") Type { ring(50, 28) } }
        Entity { id("reef1") startPosition(600, 14) color("be123c") solid(true) Type { rectangle(100, 500) } }
        Entity { id("reef2") startPosition(1200, 386) color("be123c") solid(true) Type { rectangle(100, 500) } }
        Entity { id("reef3") startPosition(1800, 14) color("be123c") solid(true) Type { rectangle(100, 500) } }
        Entity { id("eel") startPosition(900, 400) color("84cc16") Type { rectangle(90, 24) }
            Movement { speed(3) Path { waypoint(800, 600) waypoint(1100, 200) } } }
        Entity { id("urchin") startPosition(1500, 380) color("78350f") Type { star(8, 30, 16) }
            Events { OnCollision("hero") { Actions { sound("hurt") updateBar("lives", -1) setPosition("hero", 60, 420, 0) } } } }
    }

    Level {
        joystick("hero", 80, 5)
        wallBox(0, 0, 2400, 900, "020617", 14)
        Entity {
            id("hero")
            startPosition(60, 420)
            color("fbbf24")
            Type { circle(26) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnCollision("pearl1") { Actions { sound("bubble") updateBar("pearls", 1) deleteCollider() } }
                OnCollision("pearl2") { Actions { sound("bubble") updateBar("pearls", 1) deleteCollider() } }
                OnCollision("shark") { Actions { sound("hurt") updateBar("lives", -1) setPosition("hero", 60, 420, 0) } }
            }
        }
        Entity { id("pearl1") startPosition(800, 120) color("f8fafc") Type { circle(14) } }
        Entity { id("pearl2") startPosition(1600, 740) color("f8fafc") Type { circle(14) } }
        goal("chest", 2250, 60, 90, 60, "-- THE TREASURE IS YOURS --")
        Entity { id("wall1") startPosition(500, 14) color("1e293b") solid(true) Type { rectangle(80, 600) } }
        Entity { id("wall2") startPosition(1000, 286) color("1e293b") solid(true) Type { rectangle(80, 600) } }
        Entity { id("wall3") startPosition(1500, 14) color("1e293b") solid(true) Type { rectangle(80, 600) } }
        Entity { id("wall4") startPosition(2000, 286) color("1e293b") solid(true) Type { rectangle(80, 600) } }
        Entity { id("shark") startPosition(2200, 450) color("64748b") Type { triangle(50, "left") }
            Movement { speed(2.2) chase("hero") }
            Events { OnCollision("hero") { Actions { setPosition("shark", 2200, 450, 0) } } } }
        HostilePatrol("jelly", "hero", "-- STUNG --", 3, "c084fc") { waypoint(1250, 120) waypoint(1250, 780) }
    }
}
