# Bakery tray: tap a chute to drop a dough ball down that column. A ball that lands on a ball
# of the same size kneads into one twice as big, and every knead lets the balls settle again,
# so a well-stacked column folds up in a chain. Knead a 64 loaf before a column overflows.
Window {
    size(8, 11)
    background("3b2412")
    music("bg_3")
    gravity(0)

    Bar("kneads") {
        initial(0)
    }

    Entity { id("wallL") startPosition(50, 100) color("78350f") Type { rectangle(100, 900) } solid(true) }
    Entity { id("wallR") startPosition(650, 100) color("78350f") Type { rectangle(100, 900) } solid(true) }
    Entity { id("tray") startPosition(50, 100) color("92400e") Type { rectangle(700, 100) } solid(true) }

    Entity {
        id("brim")
        startPosition(150, 800)
        color("44ef4444")
        Type { rectangle(500, 100) }
        Events {
            OnCollision("ball2") { Actions { sound("lose") shake(8, 0.5) gameOver("-- THE TRAY OVERFLOWED --") } }
            OnCollision("ball4") { Actions { sound("lose") shake(8, 0.5) gameOver("-- THE TRAY OVERFLOWED --") } }
        }
    }

    Entity {
        id("ball2")
        template(true)
        startPosition(0, 0)
        color("fef3c7")
        Type { circle(45) }
        var("size", 2)
        Movement { GridMove(100, 1) { direction(0, 0) } }
        Events {
            OnElapsedTime {
                Actions {
                    sound("thud")
                    slide(0, -1, "size", "kneaded")
                }
            }
            OnSignal("kneaded") {
                Actions {
                    slide(0, -1, "size", "kneaded")
                    whenVar("size", "==", 4) { Actions { setColor("fde68a") } }
                    whenVar("size", "==", 8) { Actions { setColor("fbbf24") } }
                    whenVar("size", "==", 16) { Actions { setColor("f59e0b") } }
                    whenVar("size", "==", 32) { Actions { setColor("d97706") } }
                    whenVar("size", "==", 64) { Actions {
                        setColor("b45309")
                        sound("win")
                        burst(30, "fef3c7")
                        gameOver("-- A LOAF FIT FOR A KING --")
                    } }
                }
            }
        }
    }

    Entity {
        id("ball4")
        template(true)
        startPosition(0, 0)
        color("fde68a")
        Type { circle(45) }
        var("size", 4)
        Movement { GridMove(100, 1) { direction(0, 0) } }
        Events {
            OnElapsedTime {
                Actions {
                    sound("thud")
                    slide(0, -1, "size", "kneaded")
                }
            }
            OnSignal("kneaded") {
                Actions {
                    slide(0, -1, "size", "kneaded")
                    whenVar("size", "==", 8) { Actions { setColor("fbbf24") } }
                    whenVar("size", "==", 16) { Actions { setColor("f59e0b") } }
                    whenVar("size", "==", 32) { Actions { setColor("d97706") } }
                    whenVar("size", "==", 64) { Actions {
                        setColor("b45309")
                        sound("win")
                        burst(30, "fef3c7")
                        gameOver("-- A LOAF FIT FOR A KING --")
                    } }
                }
            }
        }
    }

    Entity {
        id("baker")
        startPosition(20, 1040)
        color("fde68a")
        Type { text("knead to 64", 36) }
        Events {
            OnSignal("kneaded") {
                Actions {
                    sound("pop")
                    updateBar("kneads", 1)
                    flash("ffffff", 0.15)
                }
            }
        }
    }

    # Five chutes along the bottom; each drops a ball (usually a 2) into its column from the top.
    Entity { id("chute1") startPosition(155, 20) color("a16207") Type { rectangle(90, 60) }
        Events { OnTapped("chute1") { Actions { sound("click") flash("ffffff", 0.1)
            randomAction { Actions { spawn("ball2", 155, 905) } Actions { spawn("ball2", 155, 905) } Actions { spawn("ball4", 155, 905) } } } } } }
    Entity { id("chute2") startPosition(255, 20) color("a16207") Type { rectangle(90, 60) }
        Events { OnTapped("chute2") { Actions { sound("click") flash("ffffff", 0.1)
            randomAction { Actions { spawn("ball2", 255, 905) } Actions { spawn("ball2", 255, 905) } Actions { spawn("ball4", 255, 905) } } } } } }
    Entity { id("chute3") startPosition(355, 20) color("a16207") Type { rectangle(90, 60) }
        Events { OnTapped("chute3") { Actions { sound("click") flash("ffffff", 0.1)
            randomAction { Actions { spawn("ball2", 355, 905) } Actions { spawn("ball2", 355, 905) } Actions { spawn("ball4", 355, 905) } } } } } }
    Entity { id("chute4") startPosition(455, 20) color("a16207") Type { rectangle(90, 60) }
        Events { OnTapped("chute4") { Actions { sound("click") flash("ffffff", 0.1)
            randomAction { Actions { spawn("ball2", 455, 905) } Actions { spawn("ball2", 455, 905) } Actions { spawn("ball4", 455, 905) } } } } } }
    Entity { id("chute5") startPosition(555, 20) color("a16207") Type { rectangle(90, 60) }
        Events { OnTapped("chute5") { Actions { sound("click") flash("ffffff", 0.1)
            randomAction { Actions { spawn("ball2", 555, 905) } Actions { spawn("ball2", 555, 905) } Actions { spawn("ball4", 555, 905) } } } } } }
}
