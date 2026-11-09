Window {
    size(6, 8)
    background("1c1917")
    music("bg_ambient")

    Bar("goals") { initial(0) }

    Level {
        wallBox(0, 200, 600, 600, "78716c", 20)
        dpad("hero", "step")

        Entity {
            id("hero")
            startPosition(100, 300)
            color("38bdf8")
            Type { circle(40) }
            Movement { GridMove(100, 0.2) { direction(0, 0) } }
        }
        Entity {
            id("ball")
            startPosition(200, 400)
            color("fafafa")
            pushable(true)
            Type { circle(40) }
            Events {
                OnCollision("net") {
                    Actions {
                        sound("coin")
                        updateBar("goals", 1)
                        nextLevel()
                    }
                }
            }
        }
        Entity {
            id("net")
            startPosition(400, 600)
            color("22c55e")
            Type { rectangle(100, 100) }
        }
        Entity {
            id("crate")
            startPosition(300, 400)
            color("a16207")
            solid(true)
            Type { rectangle(100, 100) }
        }
    }

    Level {
        wallBox(0, 200, 600, 600, "78716c", 20)
        dpad("hero", "step")

        Entity {
            id("hero")
            startPosition(100, 300)
            color("38bdf8")
            Type { circle(40) }
            Movement { GridMove(100, 0.2) { direction(0, 0) } }
        }
        Entity {
            id("ball")
            startPosition(200, 500)
            color("fafafa")
            pushable(true)
            Type { circle(40) }
            Events {
                OnCollision("net") {
                    Actions {
                        sound("win2")
                        gameOver("GOAL! BOTH PUZZLES SOLVED")
                    }
                }
            }
        }
        Entity {
            id("net")
            startPosition(400, 300)
            color("22c55e")
            Type { rectangle(100, 100) }
        }
        Entity {
            id("crate")
            startPosition(300, 500)
            color("a16207")
            pushable(true)
            Type { rectangle(100, 100) }
        }
        Entity {
            id("rock")
            startPosition(400, 500)
            color("57534e")
            solid(true)
            Type { rectangle(100, 100) }
        }
    }
}
