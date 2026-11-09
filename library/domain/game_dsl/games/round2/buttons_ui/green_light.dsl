# Tap GO only while the light is green. Early taps cost a life.
Window {
    size(8, 11)
    background("0f172a")
    music("bg_4")
    lives(3, "TOO TRIGGER-HAPPY", "lose")
    Bar("go") { initial(0) }
    Bar("hits") {
        initial(0)
        OnValue(6) { Actions { sound("win") gameOver("PERFECT TIMING") } }
    }

    Entity {
        id("light")
        startPosition(250, 650)
        color("ef4444")
        Type { circle(150) }
        Events {
            OnElapsedTime(2.0) {
                repeatCount(-1)
                Actions {
                    updateBar("go", 1)
                    setColor("22c55e")
                    delay(0.9) { Actions {
                        whenBar("go", ">", 0) { Actions { updateBar("go", -1) } }
                        setColor("ef4444")
                    } }
                }
            }
        }
    }
    Entity {
        id("status")
        startPosition(180, 1000)
        color("e2e8f0")
        Type { text("Wait for green", 48) }
    }
    Entity {
        id("goButton")
        startPosition(200, 200)
        color("38bdf8")
        Type { rectangle(400, 200) }
        Events { OnTapped("goButton") { Actions { broadcast("press") } } }
    }
    Entity {
        id("goLabel")
        startPosition(300, 270)
        color("082f49")
        Type { text("GO", 80) }
        Events { OnTapped { Actions { broadcast("press") } } }
    }
    Entity {
        id("referee")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSignal("press") {
                Actions {
                    whenBar("go", "<=", 0) { Actions {
                        sound("hurt")
                        updateBar("lives", -1)
                        setText("status", "Too early!")
                        shake(5, 0.25)
                    } }
                    whenBar("go", ">", 0) { Actions {
                        sound("coin")
                        updateBar("hits", 1)
                        updateBar("go", -1)
                        setText("status", "Nice one!")
                    } }
                }
            }
        }
    }
}
