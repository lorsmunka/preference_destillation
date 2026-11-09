Window {
    size(8, 12)
    background("1c1917")
    music("bg_3")
    gravity(0)

    Bar("time") {
        initial(25)
        OnValue(0) { Actions {
            sound("lose")
            gameOver("-- TIME IS UP --")
        } }
    }

    Level {
        wallBox(0, 0, 800, 1200, "44403c", 20)
        Entity { id("pillar") startPosition(200, 400) color("57534e")
            Type { rectangle(400, 40) } solid(true) }
        Entity { id("pillar2") startPosition(0, 800) color("57534e")
            Type { rectangle(500, 40) } solid(true) }
        Entity {
            id("racer")
            startPosition(60, 60)
            color("38bdf8")
            Type { rectangle(50, 50) }
            Movement { speed(6) Control { movementControl(0) } }
            Events {
                OnCollision("clock") {
                    Actions {
                        sound("pickup")
                        updateBar("time", 4)
                        deleteCollider()
                    }
                }
                OnCollision("line") {
                    Actions {
                        sound("win2")
                        nextLevel()
                    }
                }
            }
        }
        Entity { id("clock") startPosition(650, 450) color("facc15")
            Type { circle(24) } }
        Entity { id("line") startPosition(650, 1050) color("f8fafc")
            Type { rectangle(100, 80) } }
        Entity {
            id("ticker")
            startPosition(400, 600)
            color("00000000")
            Type { circle(1) }
            Events {
                OnElapsedTime(1) { repeatCount(-1) Actions { updateBar("time", -1) } }
            }
        }
    }

    Level {
        wallBox(0, 0, 800, 1200, "44403c", 20)
        Entity { id("barrier") startPosition(100, 300) color("57534e")
            Type { rectangle(600, 40) } solid(true) }
        Entity { id("barrier2") startPosition(100, 700) color("57534e")
            Type { rectangle(600, 40) } solid(true) }
        Entity {
            id("racer")
            startPosition(60, 60)
            color("38bdf8")
            Type { rectangle(50, 50) }
            Movement { speed(6) Control { movementControl(0) } }
            Events {
                OnCollision("clockA") {
                    Actions {
                        sound("pickup")
                        updateBar("time", 5)
                        deleteCollider()
                    }
                }
                OnCollision("clockB") {
                    Actions {
                        sound("pickup")
                        updateBar("time", 5)
                        deleteCollider()
                    }
                }
                OnCollision("line") {
                    Actions {
                        sound("win")
                        shake(8, 0.6)
                        gameOver("-- NEW RECORD --")
                    }
                }
            }
        }
        Entity { id("clockA") startPosition(80, 480) color("facc15")
            Type { circle(24) } }
        Entity { id("clockB") startPosition(680, 880) color("facc15")
            Type { circle(24) } }
        Entity { id("line") startPosition(650, 1050) color("f8fafc")
            Type { rectangle(100, 80) } }
        Entity {
            id("ticker")
            startPosition(400, 600)
            color("00000000")
            Type { circle(1) }
            Events {
                OnElapsedTime(0.8) { repeatCount(-1) Actions { updateBar("time", -1) } }
            }
        }
    }
}
