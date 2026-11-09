Window {
    size(6, 7)
    background("1c1917")
    music("bg_1")
    Bar("score") {
        initial(0)
        OnValue(8) { Actions { sound("win") gameOver("YOU HAVE SHARP EYES") } }
    }
    Bar("lives") {
        initial(3)
        OnValue(0) { Actions { sound("lose") gameOver("OUT OF LIVES") } }
    }

    Entity {
        id("label")
        startPosition(80, 600)
        color("fef3c7")
        Type { text("Tap the odd one out", 40) }
    }
    Entity {
        id("info")
        startPosition(80, 60)
        color("a8a29e")
        Type { text("8 right to win, 3 lives", 30) }
        Events {
            OnSignal("right") { Actions { setText("info", "Correct!") } }
            OnSignal("wrong") { Actions { setText("info", "That one was normal") } }
        }
    }

    Entity {
        id("director")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.3) { Actions { broadcast("newRound") } }
            OnSignal("newRound") { Actions {
                broadcast("reset")
                delay(0.1) { Actions {
                    randomAction {
                        Actions { broadcast("oddA") }
                        Actions { broadcast("oddB") }
                        Actions { broadcast("oddC") }
                        Actions { broadcast("oddD") }
                    }
                } }
            } }
        }
    }

    Entity {
        id("tileA")
        startPosition(80, 360)
        color("38bdf8")
        Type { rectangle(200, 160) }
        var("odd", 0)
        Events {
            OnTapped("tileA") { Actions {
                whenVar("odd", "==", 1) { Actions { updateBar("score", 1) broadcast("right") sound("coin") broadcast("newRound") } }
                whenVar("odd", "==", 0) { Actions { updateBar("lives", -1) broadcast("wrong") sound("buzz") shake(4, 0.2) } }
            } }
            OnSignal("reset") { Actions { setColor("38bdf8") whenVar("odd", "==", 1) { Actions { changeVar("odd", -1) } } } }
            OnSignal("oddA") { Actions { setColor("f472b6") changeVar("odd", 1) } }
        }
    }
    Entity {
        id("tileB")
        startPosition(320, 360)
        color("38bdf8")
        Type { rectangle(200, 160) }
        var("odd", 0)
        Events {
            OnTapped("tileB") { Actions {
                whenVar("odd", "==", 1) { Actions { updateBar("score", 1) broadcast("right") sound("coin") broadcast("newRound") } }
                whenVar("odd", "==", 0) { Actions { updateBar("lives", -1) broadcast("wrong") sound("buzz") shake(4, 0.2) } }
            } }
            OnSignal("reset") { Actions { setColor("38bdf8") whenVar("odd", "==", 1) { Actions { changeVar("odd", -1) } } } }
            OnSignal("oddB") { Actions { setColor("f472b6") changeVar("odd", 1) } }
        }
    }
    Entity {
        id("tileC")
        startPosition(80, 160)
        color("38bdf8")
        Type { rectangle(200, 160) }
        var("odd", 0)
        Events {
            OnTapped("tileC") { Actions {
                whenVar("odd", "==", 1) { Actions { updateBar("score", 1) broadcast("right") sound("coin") broadcast("newRound") } }
                whenVar("odd", "==", 0) { Actions { updateBar("lives", -1) broadcast("wrong") sound("buzz") shake(4, 0.2) } }
            } }
            OnSignal("reset") { Actions { setColor("38bdf8") whenVar("odd", "==", 1) { Actions { changeVar("odd", -1) } } } }
            OnSignal("oddC") { Actions { setColor("f472b6") changeVar("odd", 1) } }
        }
    }
    Entity {
        id("tileD")
        startPosition(320, 160)
        color("38bdf8")
        Type { rectangle(200, 160) }
        var("odd", 0)
        Events {
            OnTapped("tileD") { Actions {
                whenVar("odd", "==", 1) { Actions { updateBar("score", 1) broadcast("right") sound("coin") broadcast("newRound") } }
                whenVar("odd", "==", 0) { Actions { updateBar("lives", -1) broadcast("wrong") sound("buzz") shake(4, 0.2) } }
            } }
            OnSignal("reset") { Actions { setColor("38bdf8") whenVar("odd", "==", 1) { Actions { changeVar("odd", -1) } } } }
            OnSignal("oddD") { Actions { setColor("f472b6") changeVar("odd", 1) } }
        }
    }
}
