Window {
    size(8, 14)
    background("14532d")
    music("bg_1")
    gravity(0)

    Bar("slips") {
        initial(0)
        OnValue(3) { Actions { gameOver("-- TOO MANY SLIPS --") } }
    }

    Entity { id("finish") startPosition(0, 1000) color("f8fafc")
        Type { rectangle(800, 24) } }

    Entity {
        id("bike")
        startPosition(180, 40)
        color("38bdf8")
        Type { rectangle(60, 90) }
        Movement { GridMove(50, 0.1) { direction(0, 0) } }
        Events {
            OnCollision("finish") {
                Actions {
                    sound("win")
                    shake(8, 0.6)
                    gameOver("-- YOU WON THE SPRINT --")
                }
            }
            OnCollision("oil") {
                Actions {
                    sound("bump")
                    updateBar("slips", 1)
                    gridStep("bike", 0, -1)
                    gridStep("bike", 0, -1)
                }
            }
        }
    }

    Entity {
        id("rival")
        startPosition(540, 40)
        color("f43f5e")
        Type { rectangle(60, 90) }
        Movement { GridMove(50, 0.55) { direction(0, 1) } }
        Events {
            OnCollision("finish") {
                Actions {
                    sound("lose")
                    gameOver("-- THE RIVAL WON --")
                }
            }
        }
    }

    Entity {
        id("oil")
        template(true)
        startPosition(0, 0)
        color("1e1b4b")
        Type { circle(40) }
        Events {
            OnElapsedTime(3) { Actions { deleteEntity() } }
        }
    }

    Entity { id("pedal") startPosition(330, 1160) color("fbbf24")
        Type { circle(70) }
        Events {
            OnTapped("pedal") {
                Actions {
                    gridStep("bike", 0, 1)
                    sound("click")
                }
            }
        } }
    Entity { id("pedalLabel") startPosition(250, 1320) color("ffffff")
        Type { text("TAP TO PEDAL", 30) } }

    Entity {
        id("slickMaker")
        startPosition(400, 20)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(2.5) {
                repeatCount(-1)
                Actions { spawn("oil", random(40, 360), random(250, 800)) }
            }
        }
    }
}
