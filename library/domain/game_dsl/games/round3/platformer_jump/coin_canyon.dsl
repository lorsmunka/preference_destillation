Window {
    size(8, 10)
    worldSize(24, 10)
    cameraFollow("player")
    background("fed7aa")
    music("bg_1")
    gravity(850)
    lives(3, "CANYON GOT YOU")
    winAt("coins", 6, "RICH EXPLORER", "win")

    Entity {
        id("player")
        startPosition(50, 120)
        color("7c2d12")
        Type { rectangle(34, 46) }
        Movement { speed(6.5) Control { movementControl(2) } }
        Events {
            OnCollision("abyss") { Actions {
                sound("lose")
                updateBar("lives", -1)
                setPosition("player", 50, 120, 0)
            } }
        }
    }

    Entity { id("jumper") startPosition(0, 0) color("00000000")
        Type { circle(1) }
        Events { OnSwipe("up") { Actions {
            sound("jump")
            applyImpulse("player", 0, 600)
        } } } }

    Entity { id("abyss") startPosition(0, -100) color("00000000")
        Type { rectangle(2400, 40) } }

    Entity { id("rim1") startPosition(0, 0) color("b45309")
        Type { rectangle(500, 70) } solid(true) }
    Entity { id("rim2") startPosition(1900, 0) color("b45309")
        Type { rectangle(500, 70) } solid(true) }

    Entity { id("rock1") startPosition(600, 160) color("92400e")
        Type { rectangle(160, 18) } oneWay(true) }
    Entity { id("rock2") startPosition(900, 260) color("92400e")
        Type { rectangle(160, 18) } oneWay(true) }
    Entity { id("rock3") startPosition(1200, 360) color("92400e")
        Type { rectangle(160, 18) } oneWay(true) }
    Entity { id("rock4") startPosition(1450, 250) color("92400e")
        Type { rectangle(160, 18) } oneWay(true) }
    Entity { id("rock5") startPosition(1700, 150) color("92400e")
        Type { rectangle(160, 18) } oneWay(true) }

    Entity { id("lift") startPosition(520, 100) color("d97706")
        Type { rectangle(120, 18) } oneWay(true)
        Movement { speed(0.8) Path {
            waypoint(520, 100)
            waypoint(520, 300)
            dwell(0.5)
        } } }

    Entity { id("coin1") startPosition(660, 190) color("facc15")
        Type { circle(13) }
        Events { OnCollision("player") { Actions {
            sound("coin")
            updateBar("coins", 1)
            deleteEntity()
        } } } }
    Entity { id("coin2") startPosition(960, 290) color("facc15")
        Type { circle(13) }
        Events { OnCollision("player") { Actions {
            sound("coin")
            updateBar("coins", 1)
            deleteEntity()
        } } } }
    Entity { id("coin3") startPosition(1260, 390) color("facc15")
        Type { circle(13) }
        Events { OnCollision("player") { Actions {
            sound("coin")
            updateBar("coins", 1)
            deleteEntity()
        } } } }
    Entity { id("coin4") startPosition(1510, 280) color("facc15")
        Type { circle(13) }
        Events { OnCollision("player") { Actions {
            sound("coin")
            updateBar("coins", 1)
            deleteEntity()
        } } } }
    Entity { id("coin5") startPosition(1760, 180) color("facc15")
        Type { circle(13) }
        Events { OnCollision("player") { Actions {
            sound("coin")
            updateBar("coins", 1)
            deleteEntity()
        } } } }
    Entity { id("coin6") startPosition(2100, 90) color("facc15")
        Type { circle(13) }
        Events { OnCollision("player") { Actions {
            sound("coin")
            updateBar("coins", 1)
            deleteEntity()
        } } } }
}
