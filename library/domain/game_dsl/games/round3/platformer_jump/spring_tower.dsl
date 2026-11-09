Window {
    size(6, 14)
    background("1c1917")
    music("bg_4")
    gravity(800)
    lives(3, "THE TOWER WINS")

    Entity {
        id("hero")
        startPosition(270, 90)
        color("fde047")
        Type { rectangle(34, 44) }
        Movement { speed(6) Control { movementControl(2) } }
        Events {
            OnCollision("spring1") { Actions { sound("blip") applyImpulse(0, 760) } }
            OnCollision("spring2") { Actions { sound("blip") applyImpulse(0, 760) } }
            OnCollision("spring3") { Actions { sound("blip") applyImpulse(0, 840) } }
            OnCollision("bat") { Actions {
                sound("hurt")
                shake(5, 0.3)
                updateBar("lives", -1)
                setPosition("hero", 270, 90, 0)
            } }
        }
    }

    Entity { id("floor") startPosition(0, 0) color("57534e")
        Type { rectangle(600, 70) } solid(true) }

    Entity { id("spring1") startPosition(100, 70) color("22d3ee")
        Type { rectangle(60, 14) } }
    Entity { id("ledge1") startPosition(300, 330) color("a8a29e")
        Type { rectangle(240, 16) } oneWay(true) }
    Entity { id("spring2") startPosition(420, 346) color("22d3ee")
        Type { rectangle(60, 14) } }
    Entity { id("ledge2") startPosition(40, 640) color("a8a29e")
        Type { rectangle(240, 16) } oneWay(true) }
    Entity { id("spring3") startPosition(60, 656) color("22d3ee")
        Type { rectangle(60, 14) } }
    Entity { id("ledge3") startPosition(330, 960) color("a8a29e")
        Type { rectangle(240, 16) } oneWay(true) }
    Entity { id("ledge4") startPosition(40, 1200) color("a8a29e")
        Type { rectangle(240, 16) } oneWay(true) }

    Entity { id("bat") startPosition(100, 800) color("a855f7")
        Type { triangle(30, "down") }
        Movement { speed(2) Path {
            waypoint(80, 800)
            waypoint(500, 880)
        } } }

    goal("crown", 100, 1216, 80, 50, "TOP OF THE TOWER")

    Entity { id("jumpButton") startPosition(400, 8) color("0ea5e9")
        Type { rectangle(180, 50) }
        Events { OnTapped("jumpButton") { Actions {
            sound("jump")
            applyImpulse("hero", 0, 480)
        } } } }
    Entity { id("jumpLabel") startPosition(440, 18) color("082f49")
        Type { text("JUMP", 28) } }
}
