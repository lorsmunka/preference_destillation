Window {
    size(7, 11)
    background("14532d")
    music("bg_nature")
    gravity(700)
    lives(3, "WILTED")
    winAt("flowers", 4, "GARDEN BLOOMS", "win2")

    Entity {
        id("frog")
        startPosition(320, 90)
        color("a3e635")
        Type { circle(22) }
        Movement { speed(6) Control { movementControl(2) } }
        Events {
            OnCollision("pad1") { Actions { sound("bubble") applyImpulse(0, 780) } }
            OnCollision("pad2") { Actions { sound("bubble") applyImpulse(0, 780) } }
            OnCollision("pad3") { Actions { sound("bubble") applyImpulse(0, 780) } }
            OnCollision("bug") { Actions {
                sound("hurt")
                updateBar("lives", -1)
                burst(10, "ef4444")
                setPosition("frog", 320, 90, 0)
            } }
            OnCollision("beetle") { Actions {
                sound("hurt")
                updateBar("lives", -1)
                setPosition("frog", 320, 90, 0)
            } }
        }
    }

    Entity { id("lawn") startPosition(0, 0) color("166534")
        Type { rectangle(700, 70) } solid(true) }

    Entity { id("pad1") startPosition(120, 70) color("f472b6")
        Type { rectangle(80, 14) } }
    Entity { id("pad2") startPosition(500, 70) color("f472b6")
        Type { rectangle(80, 14) } }
    Entity { id("branch1") startPosition(100, 420) color("92400e")
        Type { rectangle(200, 16) } oneWay(true) }
    Entity { id("branch2") startPosition(420, 640) color("92400e")
        Type { rectangle(200, 16) } oneWay(true) }
    Entity { id("pad3") startPosition(460, 656) color("f472b6")
        Type { rectangle(80, 14) } }
    Entity { id("branch3") startPosition(80, 900) color("92400e")
        Type { rectangle(220, 16) } oneWay(true) }

    Entity { id("flower1") startPosition(160, 440) color("fb7185")
        Type { star(6, 18, 8) }
        Events { OnCollision("frog") { Actions {
            sound("sparkle")
            updateBar("flowers", 1)
            deleteEntity()
        } } } }
    Entity { id("flower2") startPosition(500, 660) color("fde047")
        Type { star(6, 18, 8) }
        Events { OnCollision("frog") { Actions {
            sound("sparkle")
            updateBar("flowers", 1)
            deleteEntity()
        } } } }
    Entity { id("flower3") startPosition(140, 920) color("c084fc")
        Type { star(6, 18, 8) }
        Events { OnCollision("frog") { Actions {
            sound("sparkle")
            updateBar("flowers", 1)
            deleteEntity()
        } } } }
    Entity { id("flower4") startPosition(620, 100) color("f8fafc")
        Type { star(6, 18, 8) }
        Events { OnCollision("frog") { Actions {
            sound("sparkle")
            updateBar("flowers", 1)
            deleteEntity()
        } } } }

    Entity { id("bug") startPosition(600, 78) color("dc2626")
        Type { circle(14) }
        Movement { speed(1.5) AutoMove { direction(-1, 0) } }
        Events { OnCollision { Actions { bounceBack() } } } }

    Entity { id("beetle") startPosition(100, 436) color("dc2626")
        Type { circle(12) }
        Movement { speed(1.2) Path {
            waypoint(100, 436)
            waypoint(280, 436)
        } } }

    Entity { id("jumpButton") startPosition(240, 12) color("65a30d")
        Type { rectangle(220, 46) }
        Events { OnTapped("jumpButton") { Actions {
            sound("jump")
            applyImpulse("frog", 0, 520)
        } } } }
}
