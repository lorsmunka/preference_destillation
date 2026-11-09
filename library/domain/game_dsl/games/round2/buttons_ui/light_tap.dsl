# Three lamps blink at random. Tap only the lit one before it fades.
Window {
    size(8, 11)
    background("18181b")
    music("bg_6")
    lives(3, "FUMBLED", "lose")
    Bar("score") {
        initial(0)
        OnValue(12) { Actions { sound("win") gameOver("QUICK FINGERS") } }
    }

    Entity {
        id("hint")
        startPosition(120, 1000)
        color("d4d4d8")
        Type { text("Tap lit lamps only", 46) }
    }
    Entity {
        id("lampOne")
        startPosition(80, 450)
        color("3f3f46")
        var("lit", 0)
        Type { rectangle(180, 260) }
        Events {
            OnTapped("lampOne") {
                Actions {
                    whenVar("lit", "<=", 0) { Actions { sound("hurt") updateBar("lives", -1) } }
                    whenVar("lit", ">", 0) { Actions {
                        sound("pop")
                        changeVar("lit", -1)
                        setColor("3f3f46")
                        updateBar("score", 1)
                    } }
                }
            }
            OnSignal("blinkOne") {
                Actions {
                    changeVar("lit", 1)
                    setColor("facc15")
                    delay(1.1) { Actions {
                        whenVar("lit", ">", 0) { Actions { changeVar("lit", -1) setColor("3f3f46") } }
                    } }
                }
            }
        }
    }
    Entity {
        id("lampTwo")
        startPosition(310, 450)
        color("3f3f46")
        var("lit", 0)
        Type { rectangle(180, 260) }
        Events {
            OnTapped("lampTwo") {
                Actions {
                    whenVar("lit", "<=", 0) { Actions { sound("hurt") updateBar("lives", -1) } }
                    whenVar("lit", ">", 0) { Actions {
                        sound("pop")
                        changeVar("lit", -1)
                        setColor("3f3f46")
                        updateBar("score", 1)
                    } }
                }
            }
            OnSignal("blinkTwo") {
                Actions {
                    changeVar("lit", 1)
                    setColor("38bdf8")
                    delay(1.1) { Actions {
                        whenVar("lit", ">", 0) { Actions { changeVar("lit", -1) setColor("3f3f46") } }
                    } }
                }
            }
        }
    }
    Entity {
        id("lampThree")
        startPosition(540, 450)
        color("3f3f46")
        var("lit", 0)
        Type { rectangle(180, 260) }
        Events {
            OnTapped("lampThree") {
                Actions {
                    whenVar("lit", "<=", 0) { Actions { sound("hurt") updateBar("lives", -1) } }
                    whenVar("lit", ">", 0) { Actions {
                        sound("pop")
                        changeVar("lit", -1)
                        setColor("3f3f46")
                        updateBar("score", 1)
                    } }
                }
            }
            OnSignal("blinkThree") {
                Actions {
                    changeVar("lit", 1)
                    setColor("f472b6")
                    delay(1.1) { Actions {
                        whenVar("lit", ">", 0) { Actions { changeVar("lit", -1) setColor("3f3f46") } }
                    } }
                }
            }
        }
    }
    Entity {
        id("conductor")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(1.0) {
                repeatCount(-1)
                Actions {
                    randomAction {
                        Actions { broadcast("blinkOne") }
                        Actions { broadcast("blinkTwo") }
                        Actions { broadcast("blinkThree") }
                    }
                }
            }
        }
    }
}
