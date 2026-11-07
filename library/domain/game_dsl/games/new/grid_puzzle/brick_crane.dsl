# Building site: the crane swings across the shaft one cell at a time and turns back at the
# walls. Tap DROP to let go of a brick; it falls, sets, and a full course of six bricks is
# carted away. Keep the wall below the red line.
Window {
    size(8, 12)
    background("1e293b")
    music("bg_2")
    gravity(0)

    Bar("courses") {
        initial(0)
    }
    winAt("courses", 8, "-- THE TOWER IS TOPPED OUT --", "win")

    Entity {
        id("wallL")
        startPosition(120, 120)
        color("64748b")
        Type { rectangle(40, 1000) }
        solid(true)
    }
    Entity {
        id("wallR")
        startPosition(640, 120)
        color("64748b")
        Type { rectangle(40, 1000) }
        solid(true)
    }
    Entity {
        id("footing")
        startPosition(120, 120)
        color("475569")
        Type { rectangle(560, 40) }
        solid(true)
    }

    Entity {
        id("redLine")
        startPosition(160, 880)
        color("33ef4444")
        Type { rectangle(480, 80) }
        Events {
            OnCollision("laid") {
                Actions {
                    sound("lose")
                    shake(8, 0.6)
                    gameOver("-- THE WALL TOPPLED --")
                }
            }
        }
    }

    Entity {
        id("crane")
        startPosition(160, 1040)
        color("facc15")
        Type { triangle(80, "down") }
        var("swing", 0)
        Movement { GridMove(80, 0.3) { direction(1, 0) } }
        Events {
            OnGridBlocked {
                Actions {
                    changeVar("swing", 1)
                    whenVar("swing", "==", 1) { Actions { setDirection("crane", -1, 0) } }
                    whenVar("swing", "==", 2) { Actions { setDirection("crane", 1, 0) changeVar("swing", -2) } }
                }
            }
            OnSignal("drop") {
                Actions {
                    sound("click")
                    spawnHere("brick")
                }
            }
        }
    }

    Entity {
        id("brick")
        template(true)
        startPosition(0, 0)
        color("f97316")
        Type { rectangle(76, 76) }
        Movement { GridMove(80, 0.1) { direction(0, -1) } }
        Events {
            OnGridBlocked {
                Actions {
                    sound("thud")
                    burst(6, "fdba74")
                    spawnHere("laid")
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("laid")
        template(true)
        startPosition(0, 0)
        color("c2410c")
        Type { rectangle(76, 76) }
        solid(true)
        Events {
            OnElapsedTime {
                Actions {
                    clearFullRows(80, 6, 160, 160, 11, "laid", "courseDone")
                }
            }
        }
    }

    Entity {
        id("foreman")
        startPosition(170, 1140)
        color("e2e8f0")
        Type { text("SIX IN A ROW, EIGHT TIMES", 32) }
        Events {
            OnSignal("courseDone") {
                Actions {
                    sound("glass")
                    shake(3, 0.2)
                    flash("4ade80", 0.3)
                    updateBar("courses", 1)
                }
            }
        }
    }

    Entity {
        id("dropPad")
        startPosition(160, 20)
        color("ca8a04")
        Type { rectangle(480, 80) }
        Events {
            OnTapped("dropPad") {
                Actions {
                    flash("fef08a", 0.1)
                    broadcast("drop")
                }
            }
        }
    }
}
