Window {
    size(5, 7)
    background("2d1b3d")
    music("bg_2")
    Bar("total") { initial(0) }
    Bar("rolls") {
        initial(0)
        OnValue(6) { Actions { sound("lose") gameOver("TOO MANY ROLLS") } }
    }

    Entity {
        id("title")
        startPosition(40, 600)
        color("f5d0fe")
        Type { text("Get 12 to 15, then stop", 36) }
    }
    Entity {
        id("face")
        startPosition(190, 400)
        color("fdf4ff")
        Type { text("?", 150) }
        Events {
            OnSignal("one") { Actions { setText("face", "1") } }
            OnSignal("two") { Actions { setText("face", "2") } }
            OnSignal("three") { Actions { setText("face", "3") } }
            OnSignal("four") { Actions { setText("face", "4") } }
            OnSignal("five") { Actions { setText("face", "5") } }
            OnSignal("six") { Actions { setText("face", "6") } }
        }
    }
    Entity {
        id("status")
        startPosition(40, 330)
        color("c084fc")
        Type { text("Roll the die (max 5 rolls)", 32) }
        Events {
            OnSignal("bust") { Actions { setText("status", "Over 15!") } }
            OnSignal("low") { Actions { setText("status", "Not enough yet") } }
        }
    }

    Entity {
        id("roll")
        startPosition(30, 100)
        color("a855f7")
        Type { rectangle(200, 140) }
        Events {
            OnTapped("roll") { Actions {
                updateBar("rolls", 1)
                sound("click")
                randomAction {
                    Actions { updateBar("total", 1) broadcast("one") }
                    Actions { updateBar("total", 2) broadcast("two") }
                    Actions { updateBar("total", 3) broadcast("three") }
                    Actions { updateBar("total", 4) broadcast("four") }
                    Actions { updateBar("total", 5) broadcast("five") }
                    Actions { updateBar("total", 6) broadcast("six") }
                }
                delay(0.1) { Actions {
                    whenBar("total", ">", 15) { Actions { broadcast("bust") sound("lose") gameOver("BUST") } }
                } }
            } }
        }
    }
    Entity {
        id("stop")
        startPosition(270, 100)
        color("ec4899")
        Type { rectangle(200, 140) }
        Events {
            OnTapped("stop") { Actions {
                whenBar("total", ">=", 12) { Actions { sound("win") gameOver("PERFECT STOP") } }
                whenBar("total", "<", 12) { Actions { sound("buzz") broadcast("low") } }
            } }
        }
    }
}
