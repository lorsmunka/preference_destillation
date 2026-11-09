# Ghost Manor: sneak through three floors of a haunted manor, take the key, unlock the door, leave before the ghosts find you.
Window {
    size(8, 12)
    background("1e1b4b")
    music("bg_3")
    lives(3, "-- THE MANOR KEEPS YOU --", "lose")

    Level {
        joystick("kid", 80, 5)
        wallBox(0, 0, 800, 1200, "312e81", 14)
        Entity { id("kid") startPosition(60, 200) color("fde047") Type { circle(24) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnCollision("ghost1") { Actions { sound("hurt") updateBar("lives", -1) setPosition("kid", 60, 200, 0) } }
                OnCollision("stairs") { Actions { sound("win2") nextLevel() } }
            } }
        Entity { id("key1") startPosition(660, 620) color("facc15") Type { rectangle(36, 36) } key("unlock1", "kid", "pickup") }
        Entity { id("door1") startPosition(14, 900) color("7c3aed") Type { rectangle(380, 30) } door("unlock1", "clang") }
        Entity { id("shelf1") startPosition(394, 900) color("44403c") solid(true) Type { rectangle(392, 30) } }
        Entity { id("table") startPosition(300, 400) color("78350f") solid(true) Type { rectangle(200, 60) } }
        Entity { id("stairs") startPosition(380, 1080) color("a5f3fc") Type { rectangle(100, 60) } }
        Entity { id("ghost1") startPosition(700, 300) color("e0e7ff") Type { circle(30) }
            Movement { speed(1.4) Path { waypoint(700, 300) waypoint(100, 700) waypoint(700, 700) } } }
    }

    Level {
        joystick("kid", 80, 5)
        wallBox(0, 0, 800, 1200, "312e81", 14)
        Entity { id("kid") startPosition(60, 200) color("fde047") Type { circle(24) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnCollision("ghost1") { Actions { sound("hurt") updateBar("lives", -1) setPosition("kid", 60, 200, 0) } }
                OnCollision("ghost2") { Actions { sound("hurt") updateBar("lives", -1) setPosition("kid", 60, 200, 0) } }
                OnCollision("stairs") { Actions { sound("win2") nextLevel() } }
            } }
        Entity { id("key1") startPosition(80, 640) color("facc15") Type { rectangle(36, 36) } key("unlock2", "kid", "pickup") }
        Entity { id("door1") startPosition(414, 960) color("7c3aed") Type { rectangle(372, 30) } door("unlock2", "clang") }
        Entity { id("shelf1") startPosition(14, 960) color("44403c") solid(true) Type { rectangle(400, 30) } }
        Entity { id("sofa") startPosition(450, 420) color("9f1239") solid(true) Type { rectangle(300, 80) } }
        Entity { id("stairs") startPosition(60, 1080) color("a5f3fc") Type { rectangle(100, 60) } }
        Entity { id("ghost1") startPosition(700, 300) color("e0e7ff") Type { circle(30) }
            Movement { speed(2) Path { waypoint(700, 250) waypoint(100, 250) } } }
        Entity { id("ghost2") startPosition(100, 800) color("c7d2fe") Type { circle(30) }
            Movement { speed(1.8) Path { waypoint(100, 800) waypoint(700, 800) dwell(1) } } }
    }

    Level {
        joystick("kid", 80, 5)
        wallBox(0, 0, 800, 1200, "312e81", 14)
        Entity { id("kid") startPosition(60, 200) color("fde047") Type { circle(24) }
            Movement { speed(0) Control { movementControl(0) } }
            Events {
                OnCollision("ghost1") { Actions { sound("hurt") updateBar("lives", -1) setPosition("kid", 60, 200, 0) } }
                OnCollision("poltergeist") { Actions { sound("hurt") updateBar("lives", -1) setPosition("kid", 60, 200, 0) setPosition("poltergeist", 700, 1000, 0) } }
                OnCollision("exit") { Actions { sound("win") shake(6, 0.4) gameOver("-- YOU ESCAPED THE MANOR --") } }
            } }
        Entity { id("key1") startPosition(700, 700) color("facc15") Type { rectangle(36, 36) } key("unlock3", "kid", "pickup") }
        Entity { id("door1") startPosition(14, 600) color("7c3aed") Type { rectangle(560, 30) } door("unlock3", "clang") }
        Entity { id("shelf1") startPosition(574, 600) color("44403c") solid(true) Type { rectangle(212, 30) } }
        Entity { id("piano") startPosition(300, 300) color("0f172a") solid(true) Type { rectangle(200, 120) } }
        Entity { id("exit") startPosition(360, 1080) color("4ade80") Type { rectangle(120, 60) } }
        Entity { id("ghost1") startPosition(700, 400) color("e0e7ff") Type { circle(30) }
            Movement { speed(2.4) Path { waypoint(700, 400) waypoint(100, 400) waypoint(100, 520) waypoint(700, 520) } } }
        Entity { id("poltergeist") startPosition(700, 1000) color("f0abfc") Type { circle(26) }
            Movement { speed(1.3) chase("kid") } }
    }
}
