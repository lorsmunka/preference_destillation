Window {
    size(6, 8)
    background("1e1b2e")
    music("bg_ambient")
    lives(6, "OUT OF TRIES", "lose")
    winAt("pairs", 3, "ALL PAIRS FOUND", "win")
    Bar("pick") {
        initial(0)
        OnValue(2) { Actions { broadcast("matchA") updateBar("pick", -2) updateBar("pairs", 1) } }
        OnValue(20) { Actions { broadcast("matchB") updateBar("pick", -20) updateBar("pairs", 1) } }
        OnValue(200) { Actions { broadcast("matchC") updateBar("pick", -200) updateBar("pairs", 1) } }
        OnValue(11) { Actions { broadcast("miss") updateBar("lives", -1) updateBar("busy", 1)
            delay(0.7) { Actions { updateBar("pick", -11) updateBar("busy", -1) } } } }
        OnValue(101) { Actions { broadcast("miss") updateBar("lives", -1) updateBar("busy", 1)
            delay(0.7) { Actions { updateBar("pick", -101) updateBar("busy", -1) } } } }
        OnValue(110) { Actions { broadcast("miss") updateBar("lives", -1) updateBar("busy", 1)
            delay(0.7) { Actions { updateBar("pick", -110) updateBar("busy", -1) } } } }
    }
    Bar("busy") { initial(0) }

    Entity {
        id("title")
        startPosition(120, 700)
        color("e9d5ff")
        Type { text("Find the pairs", 44) }
    }
    Entity {
        id("hint")
        startPosition(100, 60)
        color("a78bfa")
        Type { text("Tap two cards. 6 tries.", 32) }
        Events {
            OnSignal("matchA") { Actions { setText("hint", "A pair!") } }
            OnSignal("matchB") { Actions { setText("hint", "A pair!") } }
            OnSignal("matchC") { Actions { setText("hint", "A pair!") } }
            OnSignal("miss") { Actions { setText("hint", "No match") } }
        }
    }

    Entity {
        id("card1")
        startPosition(50, 400)
        color("475569")
        Type { rectangle(150, 150) }
        var("up", 0)
        Events {
            OnTapped("card1") { Actions {
                whenBar("busy", "==", 0) { Actions { whenVar("up", "==", 0) { Actions {
                    sound("click")
                    setColor("ef4444")
                    changeVar("up", 1)
                    updateBar("pick", 1)
                } } } }
            } }
            OnSignal("matchA") { Actions { whenVar("up", "==", 1) { Actions { changeVar("up", 1) sound("pickup") } } } }
            OnSignal("miss") { Actions { delay(0.7) { Actions { whenVar("up", "==", 1) { Actions { setColor("475569") changeVar("up", -1) } } } } } }
        }
    }
    Entity {
        id("card2")
        startPosition(225, 400)
        color("475569")
        Type { rectangle(150, 150) }
        var("up", 0)
        Events {
            OnTapped("card2") { Actions {
                whenBar("busy", "==", 0) { Actions { whenVar("up", "==", 0) { Actions {
                    sound("click")
                    setColor("22c55e")
                    changeVar("up", 1)
                    updateBar("pick", 10)
                } } } }
            } }
            OnSignal("matchB") { Actions { whenVar("up", "==", 1) { Actions { changeVar("up", 1) sound("pickup") } } } }
            OnSignal("miss") { Actions { delay(0.7) { Actions { whenVar("up", "==", 1) { Actions { setColor("475569") changeVar("up", -1) } } } } } }
        }
    }
    Entity {
        id("card3")
        startPosition(400, 400)
        color("475569")
        Type { rectangle(150, 150) }
        var("up", 0)
        Events {
            OnTapped("card3") { Actions {
                whenBar("busy", "==", 0) { Actions { whenVar("up", "==", 0) { Actions {
                    sound("click")
                    setColor("3b82f6")
                    changeVar("up", 1)
                    updateBar("pick", 100)
                } } } }
            } }
            OnSignal("matchC") { Actions { whenVar("up", "==", 1) { Actions { changeVar("up", 1) sound("pickup") } } } }
            OnSignal("miss") { Actions { delay(0.7) { Actions { whenVar("up", "==", 1) { Actions { setColor("475569") changeVar("up", -1) } } } } } }
        }
    }
    Entity {
        id("card4")
        startPosition(50, 220)
        color("475569")
        Type { rectangle(150, 150) }
        var("up", 0)
        Events {
            OnTapped("card4") { Actions {
                whenBar("busy", "==", 0) { Actions { whenVar("up", "==", 0) { Actions {
                    sound("click")
                    setColor("3b82f6")
                    changeVar("up", 1)
                    updateBar("pick", 100)
                } } } }
            } }
            OnSignal("matchC") { Actions { whenVar("up", "==", 1) { Actions { changeVar("up", 1) sound("pickup") } } } }
            OnSignal("miss") { Actions { delay(0.7) { Actions { whenVar("up", "==", 1) { Actions { setColor("475569") changeVar("up", -1) } } } } } }
        }
    }
    Entity {
        id("card5")
        startPosition(225, 220)
        color("475569")
        Type { rectangle(150, 150) }
        var("up", 0)
        Events {
            OnTapped("card5") { Actions {
                whenBar("busy", "==", 0) { Actions { whenVar("up", "==", 0) { Actions {
                    sound("click")
                    setColor("ef4444")
                    changeVar("up", 1)
                    updateBar("pick", 1)
                } } } }
            } }
            OnSignal("matchA") { Actions { whenVar("up", "==", 1) { Actions { changeVar("up", 1) sound("pickup") } } } }
            OnSignal("miss") { Actions { delay(0.7) { Actions { whenVar("up", "==", 1) { Actions { setColor("475569") changeVar("up", -1) } } } } } }
        }
    }
    Entity {
        id("card6")
        startPosition(400, 220)
        color("475569")
        Type { rectangle(150, 150) }
        var("up", 0)
        Events {
            OnTapped("card6") { Actions {
                whenBar("busy", "==", 0) { Actions { whenVar("up", "==", 0) { Actions {
                    sound("click")
                    setColor("22c55e")
                    changeVar("up", 1)
                    updateBar("pick", 10)
                } } } }
            } }
            OnSignal("matchB") { Actions { whenVar("up", "==", 1) { Actions { changeVar("up", 1) sound("pickup") } } } }
            OnSignal("miss") { Actions { delay(0.7) { Actions { whenVar("up", "==", 1) { Actions { setColor("475569") changeVar("up", -1) } } } } } }
        }
    }
}
