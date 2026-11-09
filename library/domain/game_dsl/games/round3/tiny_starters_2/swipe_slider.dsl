Window {
    size(7, 9)
    background("0f172a")
    music("bg_ambient")
    Entity {
        id("gem")
        startPosition(500, 700)
        color("22d3ee")
        Type { diamond(60, 60) }
        Movement { GridMove(100, 0.3) { direction(0, 0) } }
        Events {
            OnCollision("goal") { Actions { sound("win") burst(14, "22d3ee") gameOver("SAFE AT HOME") } }
            OnCollision("lava") { Actions { sound("death") gameOver("MELTED") } }
            OnGridBlocked { Actions { sound("bump") } }
        }
    }
    Entity {
        id("controller")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSwipe("up") { Actions { setDirection("gem", 0, 1) } }
            OnSwipe("down") { Actions { setDirection("gem", 0, -1) } }
            OnSwipe("left") { Actions { setDirection("gem", -1, 0) } }
            OnSwipe("right") { Actions { setDirection("gem", 1, 0) } }
        }
    }
    Entity {
        id("goal")
        startPosition(100, 100)
        color("facc15")
        Type { rectangle(80, 80) }
    }
    Entity {
        id("lava")
        startPosition(300, 400)
        color("dc2626")
        Type { rectangle(100, 100) }
        Movement { speed(2) Path { waypoint(300, 400) waypoint(300, 600) } }
    }
    Entity {
        id("lavaTwo")
        startPosition(100, 500)
        color("ea580c")
        Type { rectangle(100, 100) }
        Events {
            OnCollision("gem") { Actions { sound("death") gameOver("MELTED") } }
        }
    }
    Entity {
        id("help")
        startPosition(20, 830)
        color("94a3b8")
        Type { text("Swipe to slide the gem home", 28) }
    }
}
