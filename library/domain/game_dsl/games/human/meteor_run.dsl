Window {
    size(8, 12)
    background("0a0e27")
    music("bg_6")
    gravity(0)

    lives(3, "-- DESTROYED --", "lose")
    Bar("dodged") { initial(0) }

    Entity {
        id("ship")
        startPosition(352, 70)
        color("4ade80")
        Type { rectangle(96, 30) }
        Movement { speed(5.95) Control { movementControl(2) } }
    }

    Entity {
        id("meteor")
        template(true)
        startPosition(0, 0)
        color("ef4444")
        Type { triangle(54, "down") }
        Movement { speed(3.4) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("ship") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("ship", "ff0000", 0.25)
                    shake(6, 0.3)
                    burst(14, "ff6b6b")
                    deleteEntity()
                }
            }
            OnCollision {
                Actions {
                    sound("bump")
                    updateBar("dodged", 1)
                    burst(6, "ffffff")
                    deleteEntity()
                }
            }
        }
    }

    Entity { id("star1") startPosition(150, 900) color("ffffff")
        Type { star(5, 4, 2) } Movement { speed(0.85) AutoMove { direction(0, -1) } }
        Events { OnCollision {
            Actions { setPosition("star1", 150, 1180, 0) } } } }
    Entity { id("star2") startPosition(620, 600) color("a5b4fc")
        Type { star(4, 3, 1) } Movement { speed(0.85) AutoMove { direction(0, -1) } }
        Events { OnCollision {
            Actions { setPosition("star2", 620, 1180, 0) } } } }
    Entity { id("star3") startPosition(380, 1050) color("ffffff")
        Type { star(5, 3, 1) } Movement { speed(0.85) AutoMove { direction(0, -1) } }
        Events { OnCollision {
            Actions { setPosition("star3", 380, 1180, 0) } } } }

    Entity {
        id("director")
        startPosition(400, 40)
        color("00000000")
        Type { circle(1) }
        Events {
            Sequence {
                Phase(6, 1.3) {
                    Actions { spawn("meteor", random(60, 720), 1180) } }
                Phase(9, 1.0) {
                    Actions { spawn("meteor", random(60, 720), 1180) } }
                Phase(12, 0.7) {
                    Actions { spawn("meteor", random(60, 720), 1180) } }
                Phase(1, 0.6) {
                    Actions {
                        sound("win")
                        shake(5, 0.5)
                        burst(20, "fbbf24")
                        gameOver("-- RODE OUT THE STORM --")
                    } }
            }
        }
    }
}
