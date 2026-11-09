Window {
    size(6, 8)
    background("1f2937")
    music("bg_3")
    winAt("step", 4, "SAFE OPENED", "win")
    Bar("step") { initial(0) }
    Bar("ready") { initial(0) }

    Entity {
        id("code")
        startPosition(150, 640)
        color("fde047")
        Type { text("2 - 3 - 1 - 2", 60) }
        Events {
            OnElapsedTime(3) { Actions { setText("code", "? - ? - ? - ?") updateBar("ready", 1) setColor("94a3b8") } }
        }
    }
    Entity {
        id("door")
        startPosition(100, 400)
        color("475569")
        Type { rectangle(400, 160) }
        Events {
            OnSignal("good") { Actions { setColor("22c55e") sound("click") } }
        }
    }
    Entity {
        id("info")
        startPosition(60, 330)
        color("cbd5e1")
        Type { text("Memorise the code, then enter it", 30) }
        Events {
            OnElapsedTime(3) { Actions { setText("info", "Enter the code") } }
        }
    }

    Entity {
        id("key1")
        startPosition(40, 80)
        color("0ea5e9")
        Type { rectangle(160, 160) }
        var("presses", 0)
        Events {
            OnTapped("key1") { Actions {
                whenBar("ready", "==", 1) { Actions {
                    changeVar("presses", 1)
                    whenBar("step", "!=", 2) { Actions { sound("buzz") shake(5, 0.3) restart() } }
                    whenBar("step", "==", 2) { Actions { updateBar("step", 1) broadcast("good") } }
                } }
            } }
        }
    }
    Entity {
        id("key2")
        startPosition(220, 80)
        color("f97316")
        Type { rectangle(160, 160) }
        var("presses", 0)
        Events {
            OnTapped("key2") { Actions {
                whenBar("ready", "==", 1) { Actions {
                    changeVar("presses", 1)
                    whenBar("step", "==", 1) { Actions { sound("buzz") shake(5, 0.3) restart() } }
                    whenBar("step", "==", 2) { Actions { sound("buzz") shake(5, 0.3) restart() } }
                    whenBar("step", "==", 0) { Actions { updateBar("step", 1) broadcast("good") } }
                    whenBar("step", "==", 3) { Actions { updateBar("step", 1) broadcast("good") } }
                } }
            } }
        }
    }
    Entity {
        id("key3")
        startPosition(400, 80)
        color("a855f7")
        Type { rectangle(160, 160) }
        var("presses", 0)
        Events {
            OnTapped("key3") { Actions {
                whenBar("ready", "==", 1) { Actions {
                    changeVar("presses", 1)
                    whenBar("step", "!=", 1) { Actions { sound("buzz") shake(5, 0.3) restart() } }
                    whenBar("step", "==", 1) { Actions { updateBar("step", 1) broadcast("good") } }
                } }
            } }
        }
    }
    Entity { id("label1") startPosition(100, 250) color("0c4a6e") Type { text("1", 80) } }
    Entity { id("label2") startPosition(280, 250) color("7c2d12") Type { text("2", 80) } }
    Entity { id("label3") startPosition(460, 250) color("581c87") Type { text("3", 80) } }
}
