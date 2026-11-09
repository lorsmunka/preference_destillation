Window {
    size(8, 10)
    background("14532d")
    music("bg_nature")
    Entity {
        id("hole")
        startPosition(340, 860)
        color("052e16")
        Type { circle(40) }
    }
    Entity {
        id("ball")
        startPosition(100, 100)
        color("f8fafc")
        Type { circle(24) }
        Movement { speed(6) Control { movementControl(0) } }
        Events {
            OnCollision("hole") { Actions { sound("win") burst(14, "f8fafc") gameOver("HOLE IN ONE") } }
            OnCollision("bunker") { Actions { sound("thud") setPosition("ball", 100, 100, 0) } }
        }
    }
    Entity {
        id("bunker")
        startPosition(100, 400)
        color("d6b36a")
        Type { rectangle(240, 60) }
        Movement { speed(2) Path { waypoint(80, 400) waypoint(480, 400) } }
    }
    Entity {
        id("snake")
        startPosition(560, 640)
        color("ef4444")
        Type { rectangle(120, 24) }
        Movement { speed(3) Path { waypoint(560, 640) waypoint(100, 640) } }
        Events {
            OnCollision("ball") { Actions { sound("lose") gameOver("THE SNAKE GOT IT") } }
        }
    }
    Entity {
        id("flag")
        startPosition(360, 920)
        color("fde047")
        Type { text("FLAG", 30) }
    }
}
