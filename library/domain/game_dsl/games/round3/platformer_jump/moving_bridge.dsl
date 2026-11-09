Window {
    size(8, 8)
    worldSize(28, 8)
    cameraFollow("runner")
    background("0c4a6e")
    music("bg_6")
    gravity(800)
    lives(3, "SWEPT AWAY")

    Entity {
        id("runner")
        startPosition(40, 110)
        color("fbbf24")
        Type { rectangle(32, 44) }
        Movement { speed(6) Control { movementControl(2) } }
        Events {
            OnCollision("water") { Actions {
                sound("bubble")
                updateBar("lives", -1)
                setPosition("runner", 40, 110, 0)
            } }
            OnCollision("fish") { Actions {
                sound("hurt")
                updateBar("lives", -1)
                setPosition("runner", 40, 110, 0)
            } }
            OnCollision("flag") { Actions {
                sound("win")
                burst(20, "fbbf24")
                gameOver("OTHER SHORE REACHED")
            } }
        }
    }

    Entity { id("swiper") startPosition(0, 0) color("00000000")
        Type { circle(1) }
        Events { OnSwipe("up") { Actions {
            sound("jump")
            applyImpulse("runner", 0, 540)
        } } } }

    Entity { id("water") startPosition(0, -40) color("0284c7")
        Type { rectangle(2800, 60) } }

    Entity { id("bank1") startPosition(0, 0) color("78350f")
        Type { rectangle(320, 70) } solid(true) }
    Entity { id("bank2") startPosition(2480, 0) color("78350f")
        Type { rectangle(320, 70) } solid(true) }

    Entity { id("raft1") startPosition(400, 70) color("d6d3d1")
        Type { rectangle(140, 16) } oneWay(true)
        Movement { speed(1.0) Path {
            waypoint(400, 70)
            waypoint(400, 200)
            dwell(0.4)
        } } }
    Entity { id("raft2") startPosition(650, 200) color("d6d3d1")
        Type { rectangle(140, 16) } oneWay(true)
        Movement { speed(1.6) Path {
            waypoint(650, 200)
            waypoint(900, 200)
        } } }
    Entity { id("log") startPosition(1050, 120) color("92400e")
        Type { rectangle(120, 16) } oneWay(true) }
    Entity { id("raft3") startPosition(1250, 260) color("d6d3d1")
        Type { rectangle(140, 16) } oneWay(true)
        Movement { speed(1.4) Path {
            waypoint(1250, 260)
            waypoint(1250, 120)
            waypoint(1550, 120)
            waypoint(1550, 260)
        } } }
    Entity { id("log2") startPosition(1750, 200) color("92400e")
        Type { rectangle(120, 16) } oneWay(true) }
    Entity { id("raft4") startPosition(1950, 150) color("d6d3d1")
        Type { rectangle(140, 16) } oneWay(true)
        Movement { speed(1.8) Path {
            waypoint(1950, 150)
            waypoint(2300, 150)
            dwell(0.6)
        } } }

    Entity { id("fish") startPosition(1000, 20) color("f97316")
        Type { triangle(26, "up") }
        Movement { speed(2.5) Path {
            waypoint(900, 20)
            waypoint(1700, 20)
        } } }

    Entity { id("flag") startPosition(2680, 70) color("22c55e")
        Type { triangle(50, "up") } }
}
