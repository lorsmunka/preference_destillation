Window {
    size(6, 13)
    background("1c0f0a")
    music("bg_3")
    gravity(750)

    lives(2, "BURNED")

    Entity {
        id("climber")
        startPosition(250, 90)
        color("38bdf8")
        Type { rectangle(34, 46) }
        Movement { speed(6) Control { movementControl(2) } }
        Events {
            OnCollision("lava") { Actions {
                sound("hurt")
                shake(6, 0.4)
                updateBar("lives", -1)
                setPosition("climber", 250, 120, 0)
                setPosition("lava", 0, -400, 0)
            } }
            OnCollision("exit") { Actions {
                sound("win")
                burst(24, "fbbf24")
                gameOver("ESCAPED THE LAVA")
            } }
        }
    }

    Entity { id("lava") startPosition(0, -400) color("ea580c")
        Type { rectangle(600, 400) }
        Movement { speed(0.3) AutoMove { direction(0, 1) } }
        gravity(0) }

    Entity { id("rock") startPosition(0, 0) color("44403c")
        Type { rectangle(600, 60) } oneWay(true) }

    Entity { id("ledgeA") startPosition(300, 190) color("a8a29e")
        Type { rectangle(200, 16) } oneWay(true) }
    Entity { id("ledgeB") startPosition(60, 330) color("a8a29e")
        Type { rectangle(200, 16) } oneWay(true) }
    Entity { id("ledgeC") startPosition(330, 470) color("a8a29e")
        Type { rectangle(200, 16) } oneWay(true) }
    Entity { id("ledgeD") startPosition(80, 610) color("a8a29e")
        Type { rectangle(200, 16) } oneWay(true) }
    Entity { id("ledgeE") startPosition(340, 750) color("a8a29e")
        Type { rectangle(200, 16) } oneWay(true) }
    Entity { id("ledgeF") startPosition(60, 890) color("a8a29e")
        Type { rectangle(200, 16) } oneWay(true) }
    Entity { id("ledgeG") startPosition(330, 1030) color("a8a29e")
        Type { rectangle(200, 16) } oneWay(true) }
    Entity { id("ledgeH") startPosition(100, 1170) color("a8a29e")
        Type { rectangle(240, 16) } oneWay(true) }

    Entity { id("exit") startPosition(160, 1186) color("fbbf24")
        Type { star(5, 28, 12) } }

    Entity { id("leftButton") startPosition(10, 10) color("475569")
        Type { rectangle(170, 44) }
        Events { OnTapped("leftButton") { Actions {
            sound("jump")
            applyImpulse("climber", -120, 520)
        } } } }
    Entity { id("rightButton") startPosition(420, 10) color("475569")
        Type { rectangle(170, 44) }
        Events { OnTapped("rightButton") { Actions {
            sound("jump")
            applyImpulse("climber", 120, 520)
        } } } }
    Entity { id("upButton") startPosition(200, 10) color("64748b")
        Type { rectangle(200, 44) }
        Events { OnTapped("upButton") { Actions {
            sound("jump")
            applyImpulse("climber", 0, 560)
        } } } }
}
