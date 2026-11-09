Window {
    size(6, 6)
    background("111827")
    music("bg_5")
    winAt("green", 4, "ALL GREEN", "win")
    Bar("green") { initial(1) }
    Bar("taps") {
        initial(0)
        OnValue(10) { Actions { sound("lose") gameOver("OUT OF TAPS") } }
    }

    Entity {
        id("title")
        startPosition(60, 520)
        color("f3f4f6")
        Type { text("Turn every tile green", 40) }
    }
    Entity {
        id("rule")
        startPosition(60, 460)
        color("9ca3af")
        Type { text("red > green > blue > red. 10 taps.", 28) }
    }

    Entity {
        id("tile1")
        startPosition(60, 220)
        color("ef4444")
        Type { rectangle(200, 200) }
        var("state", 0)
        Events {
            OnTapped("tile1") { Actions {
                updateBar("taps", 1)
                whenVar("state", "==", 0) { Actions { setColor("22c55e") updateBar("green", 1) } }
                whenVar("state", "==", 1) { Actions { setColor("3b82f6") updateBar("green", -1) } }
                whenVar("state", "==", 2) { Actions { setColor("ef4444") changeVar("state", -3) } }
                changeVar("state", 1)
            } }
        }
    }
    Entity {
        id("tile2")
        startPosition(340, 220)
        color("3b82f6")
        Type { rectangle(200, 200) }
        var("state", 2)
        Events {
            OnTapped("tile2") { Actions {
                updateBar("taps", 1)
                whenVar("state", "==", 0) { Actions { setColor("22c55e") updateBar("green", 1) } }
                whenVar("state", "==", 1) { Actions { setColor("3b82f6") updateBar("green", -1) } }
                whenVar("state", "==", 2) { Actions { setColor("ef4444") changeVar("state", -3) } }
                changeVar("state", 1)
            } }
        }
    }
    Entity {
        id("tile3")
        startPosition(60, 0)
        color("22c55e")
        Type { rectangle(200, 200) }
        var("state", 1)
        Events {
            OnTapped("tile3") { Actions {
                updateBar("taps", 1)
                whenVar("state", "==", 0) { Actions { setColor("22c55e") updateBar("green", 1) } }
                whenVar("state", "==", 1) { Actions { setColor("3b82f6") updateBar("green", -1) } }
                whenVar("state", "==", 2) { Actions { setColor("ef4444") changeVar("state", -3) } }
                changeVar("state", 1)
            } }
        }
    }
    Entity {
        id("tile4")
        startPosition(340, 0)
        color("ef4444")
        Type { rectangle(200, 200) }
        var("state", 0)
        Events {
            OnTapped("tile4") { Actions {
                updateBar("taps", 1)
                whenVar("state", "==", 0) { Actions { setColor("22c55e") updateBar("green", 1) } }
                whenVar("state", "==", 1) { Actions { setColor("3b82f6") updateBar("green", -1) } }
                whenVar("state", "==", 2) { Actions { setColor("ef4444") changeVar("state", -3) } }
                changeVar("state", 1)
            } }
        }
    }
}
