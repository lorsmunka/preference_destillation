Window {
    size(8, 12)
    background("10131a")
    gravity(0)

    Bar("hits") {
        initial(0)
        OnValue(10) { Actions { sound("win") gameOver("-- REFLEXES: SHARP --") } }
    }
    Bar("strikes") {
        initial(0)
        OnValue(3) { Actions { sound("lose") restart() } }
    }

    Entity {
        id("prompt")
        startPosition(220, 560)
        color("e8e8e8")
        Type { text("wait for it...", 44) }
        var("ready", 0)
        Events {
            OnElapsedTime(random(1, 2.5)) {
                repeatCount(-1)
                Actions {
                    whenVar("ready", "==", 0) { Actions {
                        changeVar("ready", 1)
                        sound("beep")
                        setColor("fde047")
                        setText("prompt", "NOW -- TAP!")
                        delay(0.7) { Actions {
                            whenVar("ready", "==", 1) { Actions {
                                sound("lose")
                                setColor("ef4444")
                                flash("ff0000", 0.3)
                                shake(8, 0.4)
                                gameOver("-- TOO SLOW --") } } } } } }
                }
            }
            OnTapped {
                Actions {
                    whenVar("ready", "==", 0) { Actions {
                        sound("buzz")
                        updateBar("strikes", 1)
                        setColor("ef4444")
                        setText("prompt", "too soon!")
                        flash("ff0000", 0.25)
                        shake(6, 0.3) } }
                    whenVar("ready", "==", 1) { Actions {
                        sound("coin")
                        updateBar("hits", 1)
                        changeVar("ready", -1)
                        setColor("4ade80")
                        setText("prompt", "wait for it...")
                        flash("4ade80", 0.2)
                        burst(12, "4ade80") } }
                }
            }
        }
    }

    Entity {
        id("pulse")
        template(true)
        startPosition(0, 0)
        color("a5b4fc")
        Type { star(4, 6, 2) }
        Events {
            OnElapsedTime(0.05) {
                Actions {
                    burst(6, "a5b4fc")
                    deleteEntity()
                }
            }
        }
    }

    Wave(20, 2.0) {
        Actions { spawn("pulse", random(40, 720), random(200, 1000)) } }
    Wave(20, 1.0) {
        Actions { spawn("pulse", random(40, 720), random(200, 1000)) } }
    Wave(20, 0.4) {
        Actions { spawn("pulse", random(40, 720), random(200, 1000)) } }
}
