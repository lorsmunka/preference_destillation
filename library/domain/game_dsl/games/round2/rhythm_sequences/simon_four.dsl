# Watch the four pads light up, then tap them in the same order.
Window {
    size(8, 10)
    background("111827")
    music("bg_ambient")
    lives(3, "TOO MANY MISTAKES", "lose")
    winAt("progress", 4, "PATTERN COMPLETE", "win")
    Bar("ready") { initial(0) }

    Entity {
        id("status")
        startPosition(250, 900)
        color("e5e7eb")
        Type { text("Watch...", 44) }
    }

    Entity {
        id("pad1")
        startPosition(100, 450)
        color("b91c1c")
        Type { rectangle(280, 280) }
        Events {
            OnSignal("show1") { Actions { flash("fecaca", 0.5) sound("beep") } }
            OnTapped("pad1") { Actions {
                whenBar("ready", "==", 1) { Actions {
                    flash("fecaca", 0.2)
                    whenBar("progress", ">", 0) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", "==", 0) { Actions { updateBar("progress", 1) sound("pickup") } }
                } }
            } }
        }
    }
    Entity {
        id("pad2")
        startPosition(420, 450)
        color("1d4ed8")
        Type { rectangle(280, 280) }
        Events {
            OnSignal("show2") { Actions { flash("bfdbfe", 0.5) sound("bell") } }
            OnTapped("pad2") { Actions {
                whenBar("ready", "==", 1) { Actions {
                    flash("bfdbfe", 0.2)
                    whenBar("progress", "<", 2) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", ">", 2) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", "==", 2) { Actions { updateBar("progress", 1) sound("pickup") } }
                } }
            } }
        }
    }
    Entity {
        id("pad3")
        startPosition(100, 130)
        color("15803d")
        Type { rectangle(280, 280) }
        Events {
            OnSignal("show3") { Actions { flash("bbf7d0", 0.5) sound("blip") } }
            OnTapped("pad3") { Actions {
                whenBar("ready", "==", 1) { Actions {
                    flash("bbf7d0", 0.2)
                    whenBar("progress", "<", 1) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", ">", 1) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", "==", 1) { Actions { updateBar("progress", 1) sound("pickup") } }
                } }
            } }
        }
    }
    Entity {
        id("pad4")
        startPosition(420, 130)
        color("a16207")
        Type { rectangle(280, 280) }
        Events {
            OnSignal("show4") { Actions { flash("fef08a", 0.5) sound("click") } }
            OnTapped("pad4") { Actions {
                whenBar("ready", "==", 1) { Actions {
                    flash("fef08a", 0.2)
                    whenBar("progress", "<", 3) { Actions { updateBar("lives", -1) sound("buzz") } }
                    whenBar("progress", "==", 3) { Actions { updateBar("progress", 1) sound("pickup") } }
                } }
            } }
        }
    }

    Entity {
        id("director")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            Sequence {
                Phase(1, 1.5) { Actions { broadcast("show1") } }
                Phase(1, 0.9) { Actions { broadcast("show3") } }
                Phase(1, 0.9) { Actions { broadcast("show2") } }
                Phase(1, 0.9) { Actions { broadcast("show4") } }
                Phase(1, 1.0) { Actions {
                    setText("status", "Your turn!")
                    updateBar("ready", 1)
                } }
            }
        }
    }
}
