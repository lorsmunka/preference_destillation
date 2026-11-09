Window {
    size(8, 12)
    background("bbf7d0")
    music("bg_6")
    Bar("milk") {
        initial(0)
        OnValue(8) { Actions { sound("win2") gameOver("THE DAIRY IS OPEN") } }
    }
    Bar("full") {
        initial(0)
        OnValue(5) { Actions { broadcast("cowReady") } }
        OnValue(0) { Actions { broadcast("cowEmpty") } }
    }
    Bar("mood") {
        initial(3)
        OnValue(0) { Actions { sound("lose") gameOver("THE COW REFUSES TO COOPERATE") } }
    }

    Entity {
        id("cow")
        startPosition(200, 560)
        color("f5f5f4")
        Type { rectangle(360, 220) }
        Events {
            OnSignal("cowReady") { Actions { setColor("fde68a") } }
            OnSignal("cowEmpty") { Actions { setColor("f5f5f4") } }
            OnElapsedTime(1) { Actions { ramp("full", 0, 5, 10) } }
        }
    }
    Entity {
        id("cowHead")
        startPosition(560, 640)
        color("78716c")
        Type { circle(60) }
    }
    Entity {
        id("bucket")
        startPosition(250, 250)
        color("94a3b8")
        Type { rectangle(240, 160) }
        Events {
            OnTapped("bucket") {
                Actions {
                    whenBar("full", ">=", 5) {
                        Actions {
                            updateBar("milk", 1)
                            updateBar("full", -5)
                            sound("pop")
                            burst(10, "ffffff")
                            ramp("full", 0, 5, 10)
                        }
                    }
                    whenBar("full", "<", 5) {
                        Actions {
                            updateBar("mood", -1)
                            sound("buzz")
                            setScale("cow", 0.9)
                            delay(0.3) { Actions { setScale("cow", 1) } }
                        }
                    }
                }
            }
        }
    }
    Entity {
        id("bucketLabel")
        startPosition(290, 300)
        color("1e293b")
        Type { text("MILK", 48) }
    }
    Entity {
        id("hint")
        startPosition(60, 1100)
        color("166534")
        Type { text("Milk only when the cow glows gold", 34) }
    }
}
