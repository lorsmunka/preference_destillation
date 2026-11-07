Window {
    size(8, 12)
    background("0f172a")
    music("bg_4")
    gravity(0)

    Bar("score") {
        initial(0)
        OnValue(8) {
            Actions {
                sound("win")
                flash("head", "fbbf24", 0.5)
                shake(5, 0.4)
                burst("head", 20, "fbbf24")
                gameOver("-- FEAST COMPLETE --")
            }
        }
    }
    Bar("length") { initial(4) }

    dpad("head")

    Entity {
        id("seg")
        template(true)
        startPosition(0, 0)
        color("0891b2")
        Type { rectangle(74, 74) }
        var("life", 0)
        Events {
            OnSignal("step") {
                Actions {
                    changeVar("life", -1)
                    whenVar("life", "<=", 0) { Actions { deleteEntity() } }
                }
            }
            OnCollision("head") {
                Actions {
                    sound("lose")
                    flash("head", "ff0000", 0.4)
                    shake(7, 0.4)
                    burst(16, "0891b2")
                    gameOver("-- YOU BIT YOUR TAIL --")
                }
            }
        }
    }

    Entity { id("food1") startPosition(460, 440) color("f59e0b")
        Type { diamond(68, 68) } Events { OnCollision("head") {
            Actions { sound("coin") updateBar("score", 1) updateBar("length", 1)
                flash("head", "ffffff", 0.15) burst(12, "f59e0b")
                deleteEntity() } } } }
    Entity { id("food2") startPosition(660, 640) color("f59e0b")
        Type { diamond(68, 68) } Events { OnCollision("head") {
            Actions { sound("coin") updateBar("score", 1) updateBar("length", 1)
                flash("head", "ffffff", 0.15) burst(12, "f59e0b")
                deleteEntity() } } } }
    Entity { id("food3") startPosition(160, 740) color("f59e0b")
        Type { diamond(68, 68) } Events { OnCollision("head") {
            Actions { sound("coin") updateBar("score", 1) updateBar("length", 1)
                flash("head", "ffffff", 0.15) burst(12, "f59e0b")
                deleteEntity() } } } }
    Entity { id("food4") startPosition(560, 940) color("f59e0b")
        Type { diamond(68, 68) } Events { OnCollision("head") {
            Actions { sound("coin") updateBar("score", 1) updateBar("length", 1)
                flash("head", "ffffff", 0.15) burst(12, "f59e0b")
                deleteEntity() } } } }
    Entity { id("food5") startPosition(260, 540) color("f59e0b")
        Type { diamond(68, 68) } Events { OnCollision("head") {
            Actions { sound("coin") updateBar("score", 1) updateBar("length", 1)
                flash("head", "ffffff", 0.15) burst(12, "f59e0b")
                deleteEntity() } } } }
    Entity { id("food6") startPosition(60, 1040) color("f59e0b")
        Type { diamond(68, 68) } Events { OnCollision("head") {
            Actions { sound("coin") updateBar("score", 1) updateBar("length", 1)
                flash("head", "ffffff", 0.15) burst(12, "f59e0b")
                deleteEntity() } } } }
    Entity { id("food7") startPosition(660, 1040) color("f59e0b")
        Type { diamond(68, 68) } Events { OnCollision("head") {
            Actions { sound("coin") updateBar("score", 1) updateBar("length", 1)
                flash("head", "ffffff", 0.15) burst(12, "f59e0b")
                deleteEntity() } } } }
    Entity { id("food8") startPosition(360, 740) color("f59e0b")
        Type { diamond(68, 68) } Events { OnCollision("head") {
            Actions { sound("coin") updateBar("score", 1) updateBar("length", 1)
                flash("head", "ffffff", 0.15) burst(12, "f59e0b")
                deleteEntity() } } } }

    Entity {
        id("head")
        startPosition(60, 440)
        color("22d3ee")
        Type { circle(44) }
        Movement { GridMove(100, 0.34) { direction(1, 0) } }
        Events {
            OnGridStep {
                Actions {
                    broadcast("step")
                    spawnHere("seg", "life", "length")
                }
            }
            OnGridBlocked {
                Actions {
                    sound("lose")
                    flash("head", "ff0000", 0.4)
                    shake(7, 0.4)
                    burst(14, "22d3ee")
                    gameOver("-- HIT THE WALL --")
                }
            }
        }
    }

    Wave(20, 8) { Actions { flash("head", "67e8f9", 0.2) } }
    Wave(25, 3) { Actions { flash("head", "fbbf24", 0.25) } }
    Wave(60, 1.5) { Actions { flash("head", "ef4444", 0.2) } }
}
