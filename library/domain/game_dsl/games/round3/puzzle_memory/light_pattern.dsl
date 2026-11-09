Window {
    size(6, 8)
    background("020617")
    music("bg_ambient")
    winAt("step", 4, "PATTERN REMEMBERED", "win")
    Bar("step") { initial(0) }
    Bar("ready") { initial(0) }

    Entity {
        id("prompt")
        startPosition(70, 680)
        color("e2e8f0")
        Type { text("Watch the lamps...", 42) }
        Events {
            OnSignal("go") { Actions { setText("prompt", "Your turn!") } }
        }
    }

    Entity {
        id("conductor")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(1.0) { Actions { broadcast("flashRed") } }
            OnElapsedTime(2.0) { Actions { broadcast("flashBlue") } }
            OnElapsedTime(3.0) { Actions { broadcast("flashGreen") } }
            OnElapsedTime(4.0) { Actions { broadcast("flashBlue") } }
            OnElapsedTime(5.0) { Actions { updateBar("ready", 1) broadcast("go") sound("bell") } }
        }
    }

    Entity {
        id("redLamp")
        startPosition(60, 440)
        color("7f1d1d")
        Type { circle(100) }
        Events {
            OnSignal("flashRed") { Actions {
                setColor("ef4444") sound("blip")
                delay(0.6) { Actions { setColor("7f1d1d") } }
            } }
            OnTapped("redLamp") { Actions {
                whenBar("ready", "==", 1) { Actions {
                    whenBar("step", ">", 0) { Actions { sound("buzz") restart() } }
                    whenBar("step", "==", 0) { Actions { updateBar("step", 1) setColor("ef4444") sound("blip") delay(0.3) { Actions { setColor("7f1d1d") } } } }
                } }
            } }
        }
    }
    Entity {
        id("blueLamp")
        startPosition(340, 440)
        color("1e3a8a")
        Type { circle(100) }
        Events {
            OnSignal("flashBlue") { Actions {
                setColor("3b82f6") sound("blip")
                delay(0.6) { Actions { setColor("1e3a8a") } }
            } }
            OnTapped("blueLamp") { Actions {
                whenBar("ready", "==", 1) { Actions {
                    whenBar("step", "==", 0) { Actions { sound("buzz") restart() } }
                    whenBar("step", "==", 2) { Actions { sound("buzz") restart() } }
                    whenBar("step", "==", 1) { Actions { updateBar("step", 1) setColor("3b82f6") sound("blip") delay(0.3) { Actions { setColor("1e3a8a") } } } }
                    whenBar("step", "==", 3) { Actions { updateBar("step", 1) setColor("3b82f6") sound("blip") delay(0.3) { Actions { setColor("1e3a8a") } } } }
                } }
            } }
        }
    }
    Entity {
        id("greenLamp")
        startPosition(200, 180)
        color("14532d")
        Type { circle(100) }
        Events {
            OnSignal("flashGreen") { Actions {
                setColor("22c55e") sound("blip")
                delay(0.6) { Actions { setColor("14532d") } }
            } }
            OnTapped("greenLamp") { Actions {
                whenBar("ready", "==", 1) { Actions {
                    whenBar("step", "!=", 2) { Actions { sound("buzz") restart() } }
                    whenBar("step", "==", 2) { Actions { updateBar("step", 1) setColor("22c55e") sound("blip") delay(0.3) { Actions { setColor("14532d") } } } }
                } }
            } }
        }
    }
}
