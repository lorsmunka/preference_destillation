Window {
    size(10, 6)
    background("3f6212")
    music("bg_ambient")
    gravity(0)

    Bar("laps") {
        initial(0)
        OnValue(1) { Actions {
            sound("win")
            shake(6, 0.5)
            gameOver("-- YOUR SNAIL WINS --")
        } }
    }

    Entity { id("finish") startPosition(920, 0) color("f8fafc")
        Type { rectangle(20, 600) } }

    Entity {
        id("snail")
        startPosition(10, 400)
        color("fbbf24")
        Type { circle(34) }
        Movement { GridMove(30, 0.6) { direction(1, 0) } }
        Events {
            OnCollision("finish") { Actions { updateBar("laps", 1) } }
        }
    }
    Entity {
        id("slug")
        startPosition(10, 250)
        color("a3e635")
        Type { circle(30) }
        Movement { GridMove(30, 0.52) { direction(1, 0) } }
        Events {
            OnCollision("finish") {
                Actions {
                    sound("lose")
                    gameOver("-- THE SLUG WINS --")
                }
            }
        }
    }
    Entity {
        id("beetle")
        startPosition(10, 100)
        color("0ea5e9")
        Type { circle(28) }
        Movement { GridMove(30, 0.45) { direction(1, 0) } }
        Events {
            OnCollision("finish") {
                Actions {
                    sound("lose")
                    gameOver("-- THE BEETLE WINS --")
                }
            }
            OnElapsedTime(4) { Actions { setDirection("beetle", 0, 0) } }
            OnElapsedTime(6) { Actions { setDirection("beetle", 1, 0) } }
        }
    }

    Entity { id("lettuce") startPosition(430, 480) color("22c55e")
        Type { star(6, 50, 25) }
        Events {
            OnTapped("lettuce") {
                Actions {
                    gridStep("snail", 1, 0)
                    sound("pop")
                }
            }
        } }
}
