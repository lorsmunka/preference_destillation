# Play 16 notes on the glowing pads before the minute is up. The DEMO button plays a little tune first.
Window {
    size(8, 10)
    background("3b0764")
    music("bg_1")
    winAt("notes", 16, "WHAT A TUNE", "win")

    Entity {
        id("timeUp")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(60) { Actions {
                whenBar("notes", "<", 16) { Actions { sound("lose") gameOver("THE MUSIC BOX RAN DOWN") } }
            } }
        }
    }

    Entity {
        id("demo")
        startPosition(300, 860)
        color("fb7185")
        Type { rectangle(200, 70) }
        Events { OnTapped("demo") { Actions { broadcast("playback") sound("click") } } }
    }
    Entity {
        id("band")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSignal("playback") { Actions {
                broadcast("n1")
                delay(0.5) { Actions { broadcast("n3") } }
                delay(1.0) { Actions { broadcast("n5") } }
                delay(1.5) { Actions { broadcast("n3") } }
                delay(2.0) { Actions { broadcast("n2") } }
                delay(2.5) { Actions { broadcast("n4") } }
                delay(3.2) { Actions { broadcast("n1") } }
            } }
        }
    }

    Entity {
        id("note1")
        startPosition(60, 480)
        color("f43f5e")
        Type { circle(70) }
        Events {
            OnElapsedTime { Actions { setOpacity(0.6) } }
            OnSignal("n1") { Actions { setOpacity(1) sound("bell") delay(0.3) { Actions { setOpacity(0.6) } } } }
            OnTapped("note1") { Actions { updateBar("notes", 1) broadcast("n1") } }
        }
    }
    Entity {
        id("note2")
        startPosition(300, 480)
        color("f97316")
        Type { circle(70) }
        Events {
            OnElapsedTime { Actions { setOpacity(0.6) } }
            OnSignal("n2") { Actions { setOpacity(1) sound("blip") delay(0.3) { Actions { setOpacity(0.6) } } } }
            OnTapped("note2") { Actions { updateBar("notes", 1) broadcast("n2") } }
        }
    }
    Entity {
        id("note3")
        startPosition(540, 480)
        color("eab308")
        Type { circle(70) }
        Events {
            OnElapsedTime { Actions { setOpacity(0.6) } }
            OnSignal("n3") { Actions { setOpacity(1) sound("pickup") delay(0.3) { Actions { setOpacity(0.6) } } } }
            OnTapped("note3") { Actions { updateBar("notes", 1) broadcast("n3") } }
        }
    }
    Entity {
        id("note4")
        startPosition(180, 200)
        color("22c55e")
        Type { circle(70) }
        Events {
            OnElapsedTime { Actions { setOpacity(0.6) } }
            OnSignal("n4") { Actions { setOpacity(1) sound("sparkle") delay(0.3) { Actions { setOpacity(0.6) } } } }
            OnTapped("note4") { Actions { updateBar("notes", 1) broadcast("n4") } }
        }
    }
    Entity {
        id("note5")
        startPosition(420, 200)
        color("38bdf8")
        Type { circle(70) }
        Events {
            OnElapsedTime { Actions { setOpacity(0.6) } }
            OnSignal("n5") { Actions { setOpacity(1) sound("glass") delay(0.3) { Actions { setOpacity(0.6) } } } }
            OnTapped("note5") { Actions { updateBar("notes", 1) broadcast("n5") } }
        }
    }
}
