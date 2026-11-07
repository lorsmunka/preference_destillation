Window {
    size(7, 12)
    background("0a0e27")
    music("bg_1")
    gravity(0)

    Bar("lines") {
        initial(0)
        OnValue(10) { Actions {
            sound("win")
            shake(8, 0.6)
            burst("floor", 40, "fbbf24")
            gameOver("-- TETRIS MASTER --")
        } }
    }

    Entity { id("wallL") startPosition(110, 160) color("1e293b")
        Type { rectangle(40, 840) } solid(true) }
    Entity { id("wallR") startPosition(550, 160) color("1e293b")
        Type { rectangle(40, 840) } solid(true) }
    Entity { id("floor") startPosition(110, 160) color("1e293b")
        Type { rectangle(480, 40) } solid(true) }

    Entity { id("scorekeeper") startPosition(20, 1010) color("00000000")
        Type { circle(1) }
        Events { OnSignal("rowCleared") {
            Actions { sound("glass") updateBar("lines", 1) } } } }

    Entity { id("locked") template(true) startPosition(0, 0) color("64748b")
        Type { rectangle(40, 40) } solid(true) }

    Entity {
        id("T")
        template(true)
        startPosition(0, 0)
        color("a855f7")
        Type { tetromino(40, "T") }
        Movement { GridMove(40, 0.6) { direction(0, -1) } }
        Events {
            OnSignal("moveLeft") { Actions { gridStep(-1, 0) } }
            OnSignal("moveRight") { Actions { gridStep(1, 0) } }
            OnSignal("rotateCw") { Actions { rotate(1) } }
            OnSignal("dropFast") { Actions { gridStep(0, -1) } }
            OnGridBlocked {
                Actions {
                    burst(8, "a855f7")
                    sound("clang")
                    lock("locked")
                    clearFullRows(40, 10, 150, 200, 20, "locked",
                        "rowCleared")
                    broadcast("pieceLocked")
                }
            }
        }
    }

    Entity {
        id("I")
        template(true)
        startPosition(0, 0)
        color("06b6d4")
        Type { tetromino(40, "I") }
        Movement { GridMove(40, 0.6) { direction(0, -1) } }
        Events {
            OnSignal("moveLeft") { Actions { gridStep(-1, 0) } }
            OnSignal("moveRight") { Actions { gridStep(1, 0) } }
            OnSignal("rotateCw") { Actions { rotate(1) } }
            OnSignal("dropFast") { Actions { gridStep(0, -1) } }
            OnGridBlocked {
                Actions {
                    burst(8, "06b6d4")
                    sound("clang")
                    lock("locked")
                    clearFullRows(40, 10, 150, 200, 20, "locked",
                        "rowCleared")
                    broadcast("pieceLocked")
                }
            }
        }
    }

    Entity {
        id("O")
        template(true)
        startPosition(0, 0)
        color("fbbf24")
        Type { tetromino(40, "O") }
        Movement { GridMove(40, 0.6) { direction(0, -1) } }
        Events {
            OnSignal("moveLeft") { Actions { gridStep(-1, 0) } }
            OnSignal("moveRight") { Actions { gridStep(1, 0) } }
            OnSignal("rotateCw") { Actions { rotate(1) } }
            OnSignal("dropFast") { Actions { gridStep(0, -1) } }
            OnGridBlocked {
                Actions {
                    burst(8, "fbbf24")
                    sound("clang")
                    lock("locked")
                    clearFullRows(40, 10, 150, 200, 20, "locked",
                        "rowCleared")
                    broadcast("pieceLocked")
                }
            }
        }
    }

    Entity {
        id("L")
        template(true)
        startPosition(0, 0)
        color("f97316")
        Type { tetromino(40, "L") }
        Movement { GridMove(40, 0.6) { direction(0, -1) } }
        Events {
            OnSignal("moveLeft") { Actions { gridStep(-1, 0) } }
            OnSignal("moveRight") { Actions { gridStep(1, 0) } }
            OnSignal("rotateCw") { Actions { rotate(1) } }
            OnSignal("dropFast") { Actions { gridStep(0, -1) } }
            OnGridBlocked {
                Actions {
                    burst(8, "f97316")
                    sound("clang")
                    lock("locked")
                    clearFullRows(40, 10, 150, 200, 20, "locked",
                        "rowCleared")
                    broadcast("pieceLocked")
                }
            }
        }
    }

    Entity {
        id("J")
        template(true)
        startPosition(0, 0)
        color("3b82f6")
        Type { tetromino(40, "J") }
        Movement { GridMove(40, 0.6) { direction(0, -1) } }
        Events {
            OnSignal("moveLeft") { Actions { gridStep(-1, 0) } }
            OnSignal("moveRight") { Actions { gridStep(1, 0) } }
            OnSignal("rotateCw") { Actions { rotate(1) } }
            OnSignal("dropFast") { Actions { gridStep(0, -1) } }
            OnGridBlocked {
                Actions {
                    burst(8, "3b82f6")
                    sound("clang")
                    lock("locked")
                    clearFullRows(40, 10, 150, 200, 20, "locked",
                        "rowCleared")
                    broadcast("pieceLocked")
                }
            }
        }
    }

    Entity {
        id("S")
        template(true)
        startPosition(0, 0)
        color("4ade80")
        Type { tetromino(40, "S") }
        Movement { GridMove(40, 0.6) { direction(0, -1) } }
        Events {
            OnSignal("moveLeft") { Actions { gridStep(-1, 0) } }
            OnSignal("moveRight") { Actions { gridStep(1, 0) } }
            OnSignal("rotateCw") { Actions { rotate(1) } }
            OnSignal("dropFast") { Actions { gridStep(0, -1) } }
            OnGridBlocked {
                Actions {
                    burst(8, "4ade80")
                    sound("clang")
                    lock("locked")
                    clearFullRows(40, 10, 150, 200, 20, "locked",
                        "rowCleared")
                    broadcast("pieceLocked")
                }
            }
        }
    }

    Entity {
        id("Z")
        template(true)
        startPosition(0, 0)
        color("ef4444")
        Type { tetromino(40, "Z") }
        Movement { GridMove(40, 0.6) { direction(0, -1) } }
        Events {
            OnSignal("moveLeft") { Actions { gridStep(-1, 0) } }
            OnSignal("moveRight") { Actions { gridStep(1, 0) } }
            OnSignal("rotateCw") { Actions { rotate(1) } }
            OnSignal("dropFast") { Actions { gridStep(0, -1) } }
            OnGridBlocked {
                Actions {
                    burst(8, "ef4444")
                    sound("clang")
                    lock("locked")
                    clearFullRows(40, 10, 150, 200, 20, "locked",
                        "rowCleared")
                    broadcast("pieceLocked")
                }
            }
        }
    }

    Entity {
        id("spawner")
        startPosition(40, 40)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.05) { repeatCount(1) Actions {
                randomAction {
                    Actions { spawn("T", 270, 920) }
                    Actions { spawn("I", 230, 960) }
                    Actions { spawn("O", 270, 920) }
                    Actions { spawn("L", 270, 920) }
                    Actions { spawn("J", 270, 920) }
                    Actions { spawn("S", 270, 920) }
                    Actions { spawn("Z", 270, 920) }
                }
            } }
            OnSignal("pieceLocked") {
                Actions {
                    randomAction {
                        Actions { spawn("T", 270, 920) }
                        Actions { spawn("I", 230, 960) }
                        Actions { spawn("O", 270, 920) }
                        Actions { spawn("L", 270, 920) }
                        Actions { spawn("J", 270, 920) }
                        Actions { spawn("S", 270, 920) }
                        Actions { spawn("Z", 270, 920) }
                    }
                }
            }
        }
    }

    Entity { id("padL") startPosition(0, 0) color("3b82f6")
        Type { rectangle(170, 150) }
        Events { OnTapped("padL") {
            Actions { sound("click") broadcast("moveLeft") flash("ffffff", 0.08) } } } }
    Entity { id("padRot") startPosition(175, 0) color("8b5cf6")
        Type { rectangle(170, 150) }
        Events { OnTapped("padRot") {
            Actions { sound("blip") broadcast("rotateCw") flash("ffffff", 0.08) } } } }
    Entity { id("padDrop") startPosition(350, 0) color("10b981")
        Type { rectangle(170, 150) }
        Events { OnTapped("padDrop") {
            Actions { sound("click") broadcast("dropFast") flash("ffffff", 0.08) } } } }
    Entity { id("padR") startPosition(525, 0) color("3b82f6")
        Type { rectangle(170, 150) }
        Events { OnTapped("padR") {
            Actions { sound("click") broadcast("moveRight") flash("ffffff", 0.08) } } } }

    Entity { id("padLlbl") startPosition(50, 60) color("0a0e27")
        Type { text("LEFT", 28) } }
    Entity { id("padRotlbl") startPosition(210, 60) color("0a0e27")
        Type { text("ROTATE", 24) } }
    Entity { id("padDroplbl") startPosition(385, 60) color("0a0e27")
        Type { text("DROP", 28) } }
    Entity { id("padRlbl") startPosition(575, 60) color("0a0e27")
        Type { text("RIGHT", 26) } }
}
