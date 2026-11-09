# Watch the lights, then repeat the pattern: red, green, blue, green.
Window {
    size(8, 10)
    background("111827")
    music("bg_5")
    lives(3, "THE PATTERN WINS", "lose")
    winAt("step", 4, "PATTERN COMPLETE", "win")

    Entity {
        id("message")
        startPosition(100, 880)
        color("f3f4f6")
        Type { text("Watch the lights...", 46) }
    }
    Entity {
        id("redPad")
        startPosition(100, 480)
        color("7f1d1d")
        Type { rectangle(170, 220) }
        Events {
            OnTapped("redPad") {
                Actions {
                    flash("ef4444", 0.25)
                    whenBar("step", "!=", 0) { Actions { sound("buzz") updateBar("lives", -1) } }
                    whenBar("step", "==", 0) { Actions { sound("blip") updateBar("step", 1) } }
                }
            }
            OnSignal("showRed") { Actions { flash("ef4444", 0.4) sound("blip") } }
        }
    }
    Entity {
        id("greenPad")
        startPosition(315, 480)
        color("14532d")
        Type { rectangle(170, 220) }
        Events {
            OnTapped("greenPad") {
                Actions {
                    flash("22c55e", 0.25)
                    whenBar("step", "==", 0) { Actions { sound("buzz") updateBar("lives", -1) } }
                    whenBar("step", "==", 2) { Actions { sound("buzz") updateBar("lives", -1) } }
                    whenBar("step", "==", 1) { Actions { sound("blip") updateBar("step", 1) } }
                    whenBar("step", "==", 3) { Actions { sound("blip") updateBar("step", 1) } }
                }
            }
            OnSignal("showGreen") { Actions { flash("22c55e", 0.4) sound("blip") } }
        }
    }
    Entity {
        id("bluePad")
        startPosition(530, 480)
        color("1e3a8a")
        Type { rectangle(170, 220) }
        Events {
            OnTapped("bluePad") {
                Actions {
                    flash("3b82f6", 0.25)
                    whenBar("step", "!=", 2) { Actions { sound("buzz") updateBar("lives", -1) } }
                    whenBar("step", "==", 2) { Actions { sound("blip") updateBar("step", 1) } }
                }
            }
            OnSignal("showBlue") { Actions { flash("3b82f6", 0.4) sound("blip") } }
        }
    }
    Entity {
        id("replayButton")
        startPosition(250, 200)
        color("a78bfa")
        Type { rectangle(300, 110) }
        Events { OnTapped("replayButton") { Actions { broadcast("replay") } } }
    }
    Entity {
        id("replayLabel")
        startPosition(300, 235)
        color("2e1065")
        Type { text("REPLAY", 44) }
        Events { OnTapped { Actions { broadcast("replay") } } }
    }
    Entity {
        id("director")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.5) {
                Actions {
                    broadcast("showRed")
                    delay(0.8) { Actions { broadcast("showGreen") } }
                    delay(1.6) { Actions { broadcast("showBlue") } }
                    delay(2.4) { Actions { broadcast("showGreen") } }
                    delay(3.0) { Actions { setText("message", "Your turn!") } }
                }
            }
            OnSignal("replay") {
                Actions {
                    setText("message", "Watch again...")
                    broadcast("showRed")
                    delay(0.8) { Actions { broadcast("showGreen") } }
                    delay(1.6) { Actions { broadcast("showBlue") } }
                    delay(2.4) { Actions { broadcast("showGreen") } }
                    delay(3.0) { Actions { setText("message", "Your turn!") } }
                }
            }
        }
    }
}
