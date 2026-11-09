Window {
    size(6, 8)
    background("0b132b")
    music("bg_6")
    winAt("right", 3, "VAULT UNLOCKED", "win")
    Bar("right") { initial(0) }

    Entity {
        id("title")
        startPosition(70, 700)
        color("f1f5f9")
        Type { text("Copy the lamp pattern", 40) }
    }

    Entity {
        id("lamp1")
        startPosition(80, 520)
        color("facc15")
        Type { circle(50) }
        Events { OnElapsedTime(3) { Actions { setColor("1e293b") } } }
    }
    Entity {
        id("lamp2")
        startPosition(250, 520)
        color("334155")
        Type { circle(50) }
        Events { OnElapsedTime(3) { Actions { setColor("1e293b") } } }
    }
    Entity {
        id("lamp3")
        startPosition(420, 520)
        color("facc15")
        Type { circle(50) }
        Events { OnElapsedTime(3) { Actions { setColor("1e293b") } } }
    }

    Entity {
        id("switch1")
        startPosition(80, 260)
        color("475569")
        Type { rectangle(100, 200) }
        var("on", 0)
        Events {
            OnTapped("switch1") { Actions {
                sound("click")
                whenVar("on", "==", 0) { Actions { setColor("22d3ee") } }
                whenVar("on", "==", 1) { Actions { setColor("475569") changeVar("on", -2) } }
                changeVar("on", 1)
            } }
            OnSignal("check") { Actions { whenVar("on", "==", 1) { Actions { updateBar("right", 1) } } } }
        }
    }
    Entity {
        id("switch2")
        startPosition(250, 260)
        color("475569")
        Type { rectangle(100, 200) }
        var("on", 0)
        Events {
            OnTapped("switch2") { Actions {
                sound("click")
                whenVar("on", "==", 0) { Actions { setColor("22d3ee") } }
                whenVar("on", "==", 1) { Actions { setColor("475569") changeVar("on", -2) } }
                changeVar("on", 1)
            } }
            OnSignal("check") { Actions { whenVar("on", "==", 0) { Actions { updateBar("right", 1) } } } }
        }
    }
    Entity {
        id("switch3")
        startPosition(420, 260)
        color("475569")
        Type { rectangle(100, 200) }
        var("on", 0)
        Events {
            OnTapped("switch3") { Actions {
                sound("click")
                whenVar("on", "==", 0) { Actions { setColor("22d3ee") } }
                whenVar("on", "==", 1) { Actions { setColor("475569") changeVar("on", -2) } }
                changeVar("on", 1)
            } }
            OnSignal("check") { Actions { whenVar("on", "==", 1) { Actions { updateBar("right", 1) } } } }
        }
    }

    Entity {
        id("enter")
        startPosition(150, 60)
        color("84cc16")
        Type { rectangle(300, 120) }
        Events {
            OnTapped("enter") { Actions {
                broadcast("check")
                delay(0.3) { Actions { whenBar("right", "<", 3) { Actions { sound("buzz") shake(6, 0.3) restart() } } } }
            } }
        }
    }
}
