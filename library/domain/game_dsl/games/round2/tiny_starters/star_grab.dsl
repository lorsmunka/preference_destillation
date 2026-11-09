Window {
    size(8, 10)
    background("1e1b4b")
    gravity(0)
    winAt(8, "-- STAR COLLECTOR --", "win2")

    Entity {
        id("hero")
        startPosition(100, 100)
        color("a5b4fc")
        Type { rectangle(50, 50) }
        Movement { speed(6) Control { movementControl(0) } }
    }

    Entity {
        id("star")
        startPosition(600, 700)
        color("fde047")
        Type { star(5, 30, 13) }
        Events {
            OnCollision("hero") {
                Actions {
                    sound("sparkle")
                    updateBar(1)
                    setPosition("star", random(60, 700), random(60, 880), 0)
                }
            }
        }
    }

    Entity {
        id("rock")
        startPosition(400, 500)
        color("ef4444")
        Type { polygon(5, 28) }
        Movement { speed(3) AutoMove { direction(1, -1) } }
        Events {
            OnCollision("hero") { Actions { sound("hurt") gameOver("-- HIT BY A ROCK --") } }
            OnCollision { Actions { bounceBack() } }
        }
    }
}
