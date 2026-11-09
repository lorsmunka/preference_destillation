# The cola costs 11. Insert the right coins, never more.
Window {
    size(8, 10)
    background("0f766e")
    Bar("credit") { initial(0) }

    Entity {
        id("display")
        startPosition(100, 860)
        color("ecfeff")
        Type { text("Cola: 11. Insert coins", 46) }
    }
    Entity {
        id("machine")
        startPosition(100, 560)
        color("134e4a")
        Type { rectangle(600, 200) }
        Events {
            OnSignal("coinIn") {
                Actions {
                    whenBar("credit", "==", 11) { Actions {
                        sound("win")
                        setText("display", "Enjoy your cola!")
                        setColor("22c55e")
                        gameOver("COLD COLA")
                    } }
                    whenBar("credit", ">", 11) { Actions {
                        sound("lose")
                        shake(6, 0.4)
                        setColor("ef4444")
                        gameOver("MACHINE ATE YOUR COINS")
                    } }
                }
            }
        }
    }
    Entity {
        id("coinOne")
        startPosition(100, 220)
        color("d6d3d1")
        Type { circle(80) }
        Events {
            OnTapped("coinOne") {
                Actions {
                    sound("coin")
                    updateBar("credit", 1)
                    setText("display", "Inserted 1")
                    broadcast("coinIn")
                }
            }
        }
    }
    Entity {
        id("coinTwo")
        startPosition(340, 220)
        color("fbbf24")
        Type { circle(80) }
        Events {
            OnTapped("coinTwo") {
                Actions {
                    sound("coin")
                    updateBar("credit", 2)
                    setText("display", "Inserted 2")
                    broadcast("coinIn")
                }
            }
        }
    }
    Entity {
        id("coinFive")
        startPosition(580, 220)
        color("f97316")
        Type { circle(80) }
        Events {
            OnTapped("coinFive") {
                Actions {
                    sound("coin")
                    updateBar("credit", 5)
                    setText("display", "Inserted 5")
                    broadcast("coinIn")
                }
            }
        }
    }
}
