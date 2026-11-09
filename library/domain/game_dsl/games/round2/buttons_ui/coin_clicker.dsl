Window {
    size(8, 12)
    background("1b2a41")
    music("bg_2")
    Bar("coins") { initial(0) }
    Bar("power") { initial(0) }

    Entity {
        id("title")
        startPosition(120, 1080)
        color("f1f5f9")
        Type { text("Tap the coin to get rich", 34) }
    }
    Entity {
        id("hint")
        startPosition(120, 330)
        color("94a3b8")
        Type { text("Upgrade costs 10 coins", 30) }
    }
    Entity {
        id("coin")
        startPosition(250, 600)
        color("facc15")
        Type { circle(150) }
        Events {
            OnTapped("coin") {
                Actions {
                    sound("coin")
                    updateBar("coins", 1)
                    whenBar("power", ">=", 1) { Actions { updateBar("coins", 1) } }
                    whenBar("power", ">=", 2) { Actions { updateBar("coins", 2) } }
                    burst(5, "fde68a")
                    whenBar("coins", ">=", 80) { Actions { sound("win") gameOver("YOU ARE RICH") } }
                }
            }
        }
    }
    Entity {
        id("upgradeButton")
        startPosition(250, 150)
        color("22c55e")
        Type { rectangle(300, 110) }
        Events { OnTapped("upgradeButton") { Actions { broadcast("upgrade") } } }
    }
    Entity {
        id("upgradeLabel")
        startPosition(290, 185)
        color("052e16")
        Type { text("UPGRADE", 40) }
        Events { OnTapped { Actions { broadcast("upgrade") } } }
    }
    Entity {
        id("shop")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSignal("upgrade") {
                Actions {
                    whenBar("coins", "<", 10) { Actions { sound("buzz") setText("hint", "Not enough coins") } }
                    whenBar("coins", ">=", 10) { Actions {
                        whenBar("power", "<", 2) { Actions {
                            sound("pickup")
                            updateBar("coins", -10)
                            updateBar("power", 1)
                            setText("hint", "Stronger taps!")
                        } }
                    } }
                }
            }
        }
    }
}
