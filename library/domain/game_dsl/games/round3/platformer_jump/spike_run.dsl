Window {
    size(8, 9)
    worldSize(30, 9)
    cameraFollow("hero")
    background("312e81")
    music("bg_5")
    gravity(900)
    lives(3, "SPIKED OUT")

    Entity {
        id("hero")
        startPosition(60, 100)
        color("34d399")
        Type { rectangle(36, 48) }
        Movement { speed(7) Control { movementControl(2) } }
        Events {
            OnCollision("pit") { Actions {
                sound("lose")
                updateBar("lives", -1)
                setPosition("hero", 60, 100, 0)
            } }
        }
    }

    Entity { id("controller") startPosition(0, 0) color("00000000")
        Type { circle(1) }
        Events { OnSwipe("up") { Actions {
            sound("jump")
            applyImpulse("hero", 0, 560)
        } } } }

    Entity { id("pit") startPosition(0, -80) color("00000000")
        Type { rectangle(3000, 40) } }

    Entity { id("ground1") startPosition(0, 0) color("4338ca")
        Type { rectangle(700, 60) } solid(true) }
    Entity { id("ground2") startPosition(900, 0) color("4338ca")
        Type { rectangle(600, 60) } solid(true) }
    Entity { id("ground3") startPosition(1700, 0) color("4338ca")
        Type { rectangle(500, 60) } solid(true) }
    Entity { id("ground4") startPosition(2400, 0) color("4338ca")
        Type { rectangle(600, 60) } solid(true) }

    Entity { id("ledge1") startPosition(720, 130) color("a5b4fc")
        Type { rectangle(160, 16) } oneWay(true) }
    Entity { id("ledge2") startPosition(1500, 140) color("a5b4fc")
        Type { rectangle(180, 16) } oneWay(true) }
    Entity { id("ledge3") startPosition(2200, 150) color("a5b4fc")
        Type { rectangle(180, 16) } oneWay(true) }

    Entity { id("spikeA") startPosition(400, 60) color("f87171")
        Type { triangle(36, "up") }
        Events { OnCollision("hero") { Actions {
            sound("hurt")
            flash("hero", "ef4444", 0.3)
            updateBar("lives", -1)
            setPosition("hero", 60, 100, 0)
        } } } }
    Entity { id("spikeB") startPosition(1100, 60) color("f87171")
        Type { triangle(36, "up") }
        Events { OnCollision("hero") { Actions {
            sound("hurt")
            updateBar("lives", -1)
            setPosition("hero", 60, 100, 0)
        } } } }
    Entity { id("spikeC") startPosition(1250, 60) color("f87171")
        Type { triangle(36, "up") }
        Events { OnCollision("hero") { Actions {
            sound("hurt")
            updateBar("lives", -1)
            setPosition("hero", 60, 100, 0)
        } } } }
    Entity { id("spikeD") startPosition(1950, 60) color("f87171")
        Type { triangle(36, "up") }
        Events { OnCollision("hero") { Actions {
            sound("hurt")
            updateBar("lives", -1)
            setPosition("hero", 60, 100, 0)
        } } } }
    Entity { id("spikeE") startPosition(2600, 60) color("f87171")
        Type { triangle(36, "up") }
        Events { OnCollision("hero") { Actions {
            sound("hurt")
            updateBar("lives", -1)
            setPosition("hero", 60, 100, 0)
        } } } }

    goal("finish", 2880, 60, 80, 80, "FINISH LINE")
}
