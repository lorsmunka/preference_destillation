# Cut the wires in the right order (blue, green, red) before the clock runs out.
Window {
    size(7, 10)
    background("1a1a2e")
    music("bg_5")
    winAt("cut", 3, "BOMB DEFUSED", "win")

    Entity {
        id("hint")
        startPosition(120, 880)
        color("9ca3af")
        Type { text("Order: blue, green, red", 34) }
    }
    Entity {
        id("body")
        startPosition(100, 200)
        color("374151")
        Type { rectangle(500, 560) }
    }
    Entity {
        id("timer")
        startPosition(290, 640)
        color("ef4444")
        Type { text("9", 90) }
        Events {
            Sequence {
                Phase(1, 1) { Actions { setText("timer", "8") sound("click") } }
                Phase(1, 1) { Actions { setText("timer", "7") sound("click") } }
                Phase(1, 1) { Actions { setText("timer", "6") sound("click") } }
                Phase(1, 1) { Actions { setText("timer", "5") sound("click") } }
                Phase(1, 1) { Actions { setText("timer", "4") sound("click") } }
                Phase(1, 1) { Actions { setText("timer", "3") sound("beep") } }
                Phase(1, 1) { Actions { setText("timer", "2") sound("beep") } }
                Phase(1, 1) { Actions { setText("timer", "1") sound("beep") } }
                Phase(1, 1) { Actions { setText("timer", "0") broadcast("boom") } }
            }
        }
    }
    Entity {
        id("wireBlue")
        startPosition(160, 500)
        color("3b82f6")
        Type { rectangle(380, 50) }
        Events {
            OnTapped("wireBlue") { Actions {
                updateBar("cut", 1)
                sound("pop")
                setEnable(false)
            } }
        }
    }
    Entity {
        id("wireGreen")
        startPosition(160, 400)
        color("22c55e")
        Type { rectangle(380, 50) }
        Events {
            OnTapped("wireGreen") { Actions {
                whenBar("cut", "<", 1) { Actions { broadcast("boom") } }
                whenBar("cut", "==", 1) { Actions {
                    updateBar("cut", 1)
                    sound("pop")
                    setEnable(false)
                } }
            } }
        }
    }
    Entity {
        id("wireRed")
        startPosition(160, 300)
        color("ef4444")
        Type { rectangle(380, 50) }
        Events {
            OnTapped("wireRed") { Actions {
                whenBar("cut", "<", 2) { Actions { broadcast("boom") } }
                whenBar("cut", "==", 2) { Actions {
                    updateBar("cut", 1)
                    sound("pop")
                    setEnable(false)
                } }
            } }
        }
    }
    Entity {
        id("blast")
        startPosition(350, 480)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSignal("boom") { Actions {
                loopCount(3)
                burst(14, "f97316")
                loopCount(1)
                sound("death")
                shake(12, 0.6)
                gameOver("BOOM")
            } }
        }
    }
}
