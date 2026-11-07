# Warehouse bay. Crates fall into a bay six cells wide; swipe left or right to shift the falling
# crate, up to turn it, down to shove it three cells. The bay floor is marked out in six 3x3
# pallet spaces: a pallet space packed full is shipped out and leaves a hole. A full row of six
# is cleared too, but only shipped pallets count. Crates that settle at the dock jam the bay.
Window {
    size(7, 12)
    background("111827")
    music("bg_1")
    gravity(0)

    Bar("pallets") {
        initial(0)
    }
    winAt("pallets", 4, "-- THE TRUCK IS LOADED --", "win")
    Bar("aisles") {
        initial(0)
    }

    Entity { id("wallL") startPosition(130, 260) color("374151") Type { rectangle(40, 740) } solid(true) }
    Entity { id("wallR") startPosition(530, 260) color("374151") Type { rectangle(40, 740) } solid(true) }
    Entity { id("floor") startPosition(130, 260) color("374151") Type { rectangle(440, 40) } solid(true) }

    # Pallet spaces: two columns of three, drawn under the crates.
    Entity { id("bayMarkA") startPosition(350, 300) color("1f2937") Type { rectangle(180, 180) } }
    Entity { id("bayMarkB") startPosition(170, 480) color("1f2937") Type { rectangle(180, 180) } }
    Entity { id("bayMarkC") startPosition(350, 660) color("1f2937") Type { rectangle(180, 180) } }

    Entity {
        id("dock")
        startPosition(170, 840)
        color("33f87171")
        Type { rectangle(360, 120) }
        Events {
            OnCollision("crate") {
                Actions {
                    sound("lose")
                    shake(8, 0.6)
                    gameOver("-- THE BAY IS JAMMED --")
                }
            }
        }
    }

    Entity {
        id("crate")
        template(true)
        startPosition(0, 0)
        color("94a3b8")
        Type { rectangle(60, 60) }
        solid(true)
    }

    Entity {
        id("crateT")
        template(true)
        startPosition(0, 0)
        color("a855f7")
        Type { tetromino(60, "T") }
        Movement { GridMove(60, 0.55) { direction(0, -1) } }
        Events {
            OnSignal("shiftL") { Actions { gridStep(-1, 0) } }
            OnSignal("shiftR") { Actions { gridStep(1, 0) } }
            OnSignal("turn") { Actions { rotate(1) } }
            OnSignal("shove") {
                Actions {
                    loopCount(3)
                    gridStep(0, -1)
                }
            }
            OnGridBlocked {
                Actions {
                    sound("clang")
                    burst(8, "a855f7")
                    lock("crate")
                    clearFullBoxes(60, 3, 170, 300, 6, 9, "crate", "bayShipped")
                    clearFullRows(60, 6, 170, 300, 9, "crate", "aisleCleared")
                    broadcast("nextCrate")
                }
            }
        }
    }

    Entity {
        id("crateI")
        template(true)
        startPosition(0, 0)
        color("22d3ee")
        Type { tetromino(60, "I") }
        Movement { GridMove(60, 0.55) { direction(0, -1) } }
        Events {
            OnSignal("shiftL") { Actions { gridStep(-1, 0) } }
            OnSignal("shiftR") { Actions { gridStep(1, 0) } }
            OnSignal("turn") { Actions { rotate(1) } }
            OnSignal("shove") {
                Actions {
                    loopCount(3)
                    gridStep(0, -1)
                }
            }
            OnGridBlocked {
                Actions {
                    sound("clang")
                    burst(8, "22d3ee")
                    lock("crate")
                    clearFullBoxes(60, 3, 170, 300, 6, 9, "crate", "bayShipped")
                    clearFullRows(60, 6, 170, 300, 9, "crate", "aisleCleared")
                    broadcast("nextCrate")
                }
            }
        }
    }

    Entity {
        id("crateO")
        template(true)
        startPosition(0, 0)
        color("facc15")
        Type { tetromino(60, "O") }
        Movement { GridMove(60, 0.55) { direction(0, -1) } }
        Events {
            OnSignal("shiftL") { Actions { gridStep(-1, 0) } }
            OnSignal("shiftR") { Actions { gridStep(1, 0) } }
            OnSignal("turn") { Actions { rotate(1) } }
            OnSignal("shove") {
                Actions {
                    loopCount(3)
                    gridStep(0, -1)
                }
            }
            OnGridBlocked {
                Actions {
                    sound("clang")
                    burst(8, "facc15")
                    lock("crate")
                    clearFullBoxes(60, 3, 170, 300, 6, 9, "crate", "bayShipped")
                    clearFullRows(60, 6, 170, 300, 9, "crate", "aisleCleared")
                    broadcast("nextCrate")
                }
            }
        }
    }

    Entity {
        id("crateL")
        template(true)
        startPosition(0, 0)
        color("fb923c")
        Type { tetromino(60, "L") }
        Movement { GridMove(60, 0.55) { direction(0, -1) } }
        Events {
            OnSignal("shiftL") { Actions { gridStep(-1, 0) } }
            OnSignal("shiftR") { Actions { gridStep(1, 0) } }
            OnSignal("turn") { Actions { rotate(1) } }
            OnSignal("shove") {
                Actions {
                    loopCount(3)
                    gridStep(0, -1)
                }
            }
            OnGridBlocked {
                Actions {
                    sound("clang")
                    burst(8, "fb923c")
                    lock("crate")
                    clearFullBoxes(60, 3, 170, 300, 6, 9, "crate", "bayShipped")
                    clearFullRows(60, 6, 170, 300, 9, "crate", "aisleCleared")
                    broadcast("nextCrate")
                }
            }
        }
    }

    Entity {
        id("crateJ")
        template(true)
        startPosition(0, 0)
        color("60a5fa")
        Type { tetromino(60, "J") }
        Movement { GridMove(60, 0.55) { direction(0, -1) } }
        Events {
            OnSignal("shiftL") { Actions { gridStep(-1, 0) } }
            OnSignal("shiftR") { Actions { gridStep(1, 0) } }
            OnSignal("turn") { Actions { rotate(1) } }
            OnSignal("shove") {
                Actions {
                    loopCount(3)
                    gridStep(0, -1)
                }
            }
            OnGridBlocked {
                Actions {
                    sound("clang")
                    burst(8, "60a5fa")
                    lock("crate")
                    clearFullBoxes(60, 3, 170, 300, 6, 9, "crate", "bayShipped")
                    clearFullRows(60, 6, 170, 300, 9, "crate", "aisleCleared")
                    broadcast("nextCrate")
                }
            }
        }
    }

    Entity {
        id("crateS")
        template(true)
        startPosition(0, 0)
        color("4ade80")
        Type { tetromino(60, "S") }
        Movement { GridMove(60, 0.55) { direction(0, -1) } }
        Events {
            OnSignal("shiftL") { Actions { gridStep(-1, 0) } }
            OnSignal("shiftR") { Actions { gridStep(1, 0) } }
            OnSignal("turn") { Actions { rotate(1) } }
            OnSignal("shove") {
                Actions {
                    loopCount(3)
                    gridStep(0, -1)
                }
            }
            OnGridBlocked {
                Actions {
                    sound("clang")
                    burst(8, "4ade80")
                    lock("crate")
                    clearFullBoxes(60, 3, 170, 300, 6, 9, "crate", "bayShipped")
                    clearFullRows(60, 6, 170, 300, 9, "crate", "aisleCleared")
                    broadcast("nextCrate")
                }
            }
        }
    }

    Entity {
        id("crateZ")
        template(true)
        startPosition(0, 0)
        color("f87171")
        Type { tetromino(60, "Z") }
        Movement { GridMove(60, 0.55) { direction(0, -1) } }
        Events {
            OnSignal("shiftL") { Actions { gridStep(-1, 0) } }
            OnSignal("shiftR") { Actions { gridStep(1, 0) } }
            OnSignal("turn") { Actions { rotate(1) } }
            OnSignal("shove") {
                Actions {
                    loopCount(3)
                    gridStep(0, -1)
                }
            }
            OnGridBlocked {
                Actions {
                    sound("clang")
                    burst(8, "f87171")
                    lock("crate")
                    clearFullBoxes(60, 3, 170, 300, 6, 9, "crate", "bayShipped")
                    clearFullRows(60, 6, 170, 300, 9, "crate", "aisleCleared")
                    broadcast("nextCrate")
                }
            }
        }
    }

    Entity {
        id("dispatcher")
        startPosition(20, 1120)
        color("94a3b8")
        Type { text("PACK THE PALLETS", 36) }
        Events {
            OnElapsedTime(0.5) {
                Actions { broadcast("nextCrate") }
            }
            OnSignal("nextCrate") {
                Actions {
                    randomAction {
                        Actions { spawn("crateT", 290, 840) }
                        Actions { spawn("crateI", 290, 840) }
                        Actions { spawn("crateO", 290, 840) }
                        Actions { spawn("crateL", 290, 840) }
                        Actions { spawn("crateJ", 290, 840) }
                        Actions { spawn("crateS", 290, 840) }
                        Actions { spawn("crateZ", 290, 840) }
                    }
                }
            }
            OnSignal("bayShipped") {
                Actions {
                    sound("win2")
                    shake(4, 0.3)
                    flash("4ade80", 0.4)
                    updateBar("pallets", 1)
                }
            }
            OnSignal("aisleCleared") {
                Actions {
                    sound("glass")
                    updateBar("aisles", 1)
                }
            }
        }
    }

    Entity {
        id("forklift")
        startPosition(170, 60)
        color("fbbf24")
        Type { text("swipe: move  up: turn  down: shove", 26) }
        Events {
            OnSwipe("left") { Actions { sound("click") broadcast("shiftL") } }
            OnSwipe("right") { Actions { sound("click") broadcast("shiftR") } }
            OnSwipe("up") { Actions { sound("blip") broadcast("turn") } }
            OnSwipe("down") { Actions { sound("click") broadcast("shove") } }
        }
    }
}
