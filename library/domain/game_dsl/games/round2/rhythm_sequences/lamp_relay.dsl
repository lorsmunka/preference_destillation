# A light runs down the row of lamps and gets quicker. Tap each lamp while it is lit.
Window {
    size(8, 8)
    background("1b1b2f")
    music("bg_ambient")
    lives(3, "THE RELAY BROKE", "lose")
    winAt("score", 15, "RELAY COMPLETE", "win")

    Entity {
        id("starter")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events { OnElapsedTime(1.5) { Actions { broadcast("lit1") } } }
    }

    Entity {
        id("lamp1")
        startPosition(40, 340)
        color("3f3f46")
        Type { circle(60) }
        var("lit", 0)
        Events {
            OnSignal("lit1") { Actions {
                setColor("fde047")
                changeVar("lit", 1)
                delay(0.9) { Actions {
                    whenVar("lit", ">", 0) { Actions { changeVar("lit", -1) setColor("3f3f46") updateBar("lives", -1) sound("hurt") } }
                    broadcast("lit2")
                } }
            } }
            OnTapped("lamp1") { Actions {
                whenVar("lit", ">", 0) { Actions { changeVar("lit", -1) setColor("3f3f46") updateBar("score", 1) sound("pickup") } }
            } }
        }
    }
    Entity {
        id("lamp2")
        startPosition(190, 340)
        color("3f3f46")
        Type { circle(60) }
        var("lit", 0)
        Events {
            OnSignal("lit2") { Actions {
                setColor("fb923c")
                changeVar("lit", 1)
                delay(0.8) { Actions {
                    whenVar("lit", ">", 0) { Actions { changeVar("lit", -1) setColor("3f3f46") updateBar("lives", -1) sound("hurt") } }
                    broadcast("lit3")
                } }
            } }
            OnTapped("lamp2") { Actions {
                whenVar("lit", ">", 0) { Actions { changeVar("lit", -1) setColor("3f3f46") updateBar("score", 1) sound("pickup") } }
            } }
        }
    }
    Entity {
        id("lamp3")
        startPosition(340, 340)
        color("3f3f46")
        Type { circle(60) }
        var("lit", 0)
        Events {
            OnSignal("lit3") { Actions {
                setColor("f472b6")
                changeVar("lit", 1)
                delay(0.7) { Actions {
                    whenVar("lit", ">", 0) { Actions { changeVar("lit", -1) setColor("3f3f46") updateBar("lives", -1) sound("hurt") } }
                    broadcast("lit4")
                } }
            } }
            OnTapped("lamp3") { Actions {
                whenVar("lit", ">", 0) { Actions { changeVar("lit", -1) setColor("3f3f46") updateBar("score", 1) sound("pickup") } }
            } }
        }
    }
    Entity {
        id("lamp4")
        startPosition(490, 340)
        color("3f3f46")
        Type { circle(60) }
        var("lit", 0)
        Events {
            OnSignal("lit4") { Actions {
                setColor("60a5fa")
                changeVar("lit", 1)
                delay(0.6) { Actions {
                    whenVar("lit", ">", 0) { Actions { changeVar("lit", -1) setColor("3f3f46") updateBar("lives", -1) sound("hurt") } }
                    broadcast("lit5")
                } }
            } }
            OnTapped("lamp4") { Actions {
                whenVar("lit", ">", 0) { Actions { changeVar("lit", -1) setColor("3f3f46") updateBar("score", 1) sound("pickup") } }
            } }
        }
    }
    Entity {
        id("lamp5")
        startPosition(640, 340)
        color("3f3f46")
        Type { circle(60) }
        var("lit", 0)
        Events {
            OnSignal("lit5") { Actions {
                setColor("4ade80")
                changeVar("lit", 1)
                delay(0.5) { Actions {
                    whenVar("lit", ">", 0) { Actions { changeVar("lit", -1) setColor("3f3f46") updateBar("lives", -1) sound("hurt") } }
                    broadcast("lit1")
                } }
            } }
            OnTapped("lamp5") { Actions {
                whenVar("lit", ">", 0) { Actions { changeVar("lit", -1) setColor("3f3f46") updateBar("score", 1) sound("pickup") } }
            } }
        }
    }
}
