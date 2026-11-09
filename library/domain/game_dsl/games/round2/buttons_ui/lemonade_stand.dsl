Window {
    size(8, 12)
    background("fef9c3")
    music("bg_nature")
    Bar("lemons") { initial(0) }
    Bar("cash") { initial(0) }
    Bar("sign") { initial(0) }

    Entity {
        id("heading")
        startPosition(100, 1090)
        color("854d0e")
        Type { text("Lemonade Stand", 56) }
    }
    Entity {
        id("message")
        startPosition(100, 960)
        color("a16207")
        Type { text("Squeeze, then sell. Reach 40 cash!", 32) }
    }
    Entity {
        id("squeezeButton")
        startPosition(150, 700)
        color("facc15")
        Type { rectangle(500, 140) }
        Events { OnTapped("squeezeButton") { Actions { broadcast("squeeze") } } }
    }
    Entity {
        id("squeezeLabel")
        startPosition(280, 745)
        color("713f12")
        Type { text("SQUEEZE", 50) }
        Events { OnTapped { Actions { broadcast("squeeze") } } }
    }
    Entity {
        id("sellButton")
        startPosition(150, 480)
        color("4ade80")
        Type { rectangle(500, 140) }
        Events { OnTapped("sellButton") { Actions { broadcast("sell") } } }
    }
    Entity {
        id("sellLabel")
        startPosition(340, 525)
        color("14532d")
        Type { text("SELL", 50) }
        Events { OnTapped { Actions { broadcast("sell") } } }
    }
    Entity {
        id("signButton")
        startPosition(150, 260)
        color("fb923c")
        Type { rectangle(500, 140) }
        Events { OnTapped("signButton") { Actions { broadcast("buySign") } } }
    }
    Entity {
        id("signLabel")
        startPosition(210, 305)
        color("7c2d12")
        Type { text("BUY SIGN (8)", 50) }
        Events { OnTapped { Actions { broadcast("buySign") } } }
    }
    Entity {
        id("stand")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSignal("squeeze") {
                Actions {
                    sound("bubble")
                    updateBar("lemons", 1)
                    setText("message", "Fresh lemonade!")
                }
            }
            OnSignal("sell") {
                Actions {
                    whenBar("lemons", "<", 1) { Actions { sound("buzz") setText("message", "No lemonade to sell") } }
                    whenBar("lemons", ">=", 1) { Actions {
                        sound("coin")
                        updateBar("lemons", -1)
                        updateBar("cash", 2)
                        whenBar("sign", ">=", 1) { Actions { updateBar("cash", 1) } }
                        setText("message", "Sold a cup")
                    } }
                    whenBar("cash", ">=", 40) { Actions { sound("win") gameOver("STAND IS A HIT") } }
                }
            }
            OnSignal("buySign") {
                Actions {
                    whenBar("sign", "<", 1) { Actions {
                        whenBar("cash", ">=", 8) { Actions {
                            sound("pickup")
                            updateBar("cash", -8)
                            updateBar("sign", 1)
                            setText("message", "Sign up: +1 per cup")
                        } }
                    } }
                }
            }
        }
    }
}
