Window {
    size(8, 12)
    background("14532d")
    music("bg_3")
    Bar("water") {
        initial(4)
        OnValue(0) { Actions { sound("lose") gameOver("THE GARDEN WITHERED") } }
    }
    Bar("bloom") {
        initial(0)
        OnValue(9) { Actions { sound("win") gameOver("THE GARDEN IS IN FULL BLOOM") } }
    }

    Entity {
        id("sun")
        startPosition(60, 1000)
        color("fde047")
        Type { star(8, 70, 40) }
        Events {
            OnElapsedTime(4) {
                repeatCount(-1)
                Actions { updateBar("water", -1) updateBar("bloom", 1) setScale("sun", 1.2) delay(0.3) { Actions { setScale("sun", 1) } } }
            }
        }
    }
    Entity {
        id("rose")
        startPosition(100, 500)
        color("166534")
        Type { circle(50) }
        Events {
            OnSignal("watered") { Actions { setColor("f43f5e") setScale(1.3) } }
            OnTapped("rose") { Actions { setColor("166534") setScale(1) } }
        }
    }
    Entity {
        id("daisy")
        startPosition(340, 500)
        color("166534")
        Type { circle(50) }
        Events {
            OnSignal("watered") { Actions { setColor("fafafa") setScale(1.3) } }
            OnTapped("daisy") { Actions { setColor("166534") setScale(1) } }
        }
    }
    Entity {
        id("lavender")
        startPosition(580, 500)
        color("166534")
        Type { circle(50) }
        Events {
            OnSignal("watered") { Actions { setColor("a78bfa") setScale(1.3) } }
            OnTapped("lavender") { Actions { setColor("166534") setScale(1) } }
        }
    }
    Entity {
        id("wateringCan")
        startPosition(280, 140)
        color("38bdf8")
        Type { rectangle(240, 140) }
        Events {
            OnTapped("wateringCan") {
                Actions {
                    updateBar("water", 1)
                    broadcast("watered")
                    sound("bubble")
                    burst(10, "7dd3fc")
                }
            }
        }
    }
    Entity {
        id("canLabel")
        startPosition(320, 190)
        color("0c4a6e")
        Type { text("WATER", 40) }
    }
}
