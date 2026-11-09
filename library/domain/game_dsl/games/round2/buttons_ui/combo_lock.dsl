# A 3-dial lock. Each dial shows 0-3 and wraps around. The code is 2-0-3.
Window {
    size(8, 12)
    background("1e293b")
    music("bg_ambient")
    lives(3, "THE LOCK JAMMED", "lose")
    Bar("d1") { initial(0) }
    Bar("d2") { initial(0) }
    Bar("d3") { initial(0) }
    Bar("matches") { initial(0) }

    Entity {
        id("message")
        startPosition(100, 1040)
        color("e2e8f0")
        Type { text("Tap the digits to turn them", 40) }
    }
    Entity {
        id("hint")
        startPosition(100, 940)
        color("fbbf24")
        Type { text("Hint: the code adds up to 5", 36) }
        Events { OnElapsedTime { Actions { setVisible(false) } } }
    }
    Entity {
        id("lockLamp")
        startPosition(330, 830)
        color("ef4444")
        Type { circle(70) }
        Events { OnSignal("unlocked") { Actions { setColor("22c55e") } } }
    }

    Entity {
        id("dialOne")
        startPosition(120, 480)
        color("38bdf8")
        Type { text("0", 200) }
        Events {
            OnTapped("dialOne") {
                Actions {
                    sound("click")
                    updateBar("d1", 1)
                    whenBar("d1", ">", 3) { Actions { updateBar("d1", -4) } }
                    whenBar("d1", "==", 0) { Actions { setText("dialOne", "0") } }
                    whenBar("d1", "==", 1) { Actions { setText("dialOne", "1") } }
                    whenBar("d1", "==", 2) { Actions { setText("dialOne", "2") } }
                    whenBar("d1", "==", 3) { Actions { setText("dialOne", "3") } }
                }
            }
            OnSignal("tryCode") {
                Actions {
                    whenBar("d1", "==", 2) { Actions {
                        updateBar("matches", 1)
                        delay(0.5) { Actions { updateBar("matches", -1) } }
                    } }
                }
            }
        }
    }
    Entity {
        id("dialTwo")
        startPosition(350, 480)
        color("a78bfa")
        Type { text("0", 200) }
        Events {
            OnTapped("dialTwo") {
                Actions {
                    sound("click")
                    updateBar("d2", 1)
                    whenBar("d2", ">", 3) { Actions { updateBar("d2", -4) } }
                    whenBar("d2", "==", 0) { Actions { setText("dialTwo", "0") } }
                    whenBar("d2", "==", 1) { Actions { setText("dialTwo", "1") } }
                    whenBar("d2", "==", 2) { Actions { setText("dialTwo", "2") } }
                    whenBar("d2", "==", 3) { Actions { setText("dialTwo", "3") } }
                }
            }
            OnSignal("tryCode") {
                Actions {
                    whenBar("d2", "==", 0) { Actions {
                        updateBar("matches", 1)
                        delay(0.5) { Actions { updateBar("matches", -1) } }
                    } }
                }
            }
        }
    }
    Entity {
        id("dialThree")
        startPosition(580, 480)
        color("f472b6")
        Type { text("0", 200) }
        Events {
            OnTapped("dialThree") {
                Actions {
                    sound("click")
                    updateBar("d3", 1)
                    whenBar("d3", ">", 3) { Actions { updateBar("d3", -4) } }
                    whenBar("d3", "==", 0) { Actions { setText("dialThree", "0") } }
                    whenBar("d3", "==", 1) { Actions { setText("dialThree", "1") } }
                    whenBar("d3", "==", 2) { Actions { setText("dialThree", "2") } }
                    whenBar("d3", "==", 3) { Actions { setText("dialThree", "3") } }
                }
            }
            OnSignal("tryCode") {
                Actions {
                    whenBar("d3", "==", 3) { Actions {
                        updateBar("matches", 1)
                        delay(0.5) { Actions { updateBar("matches", -1) } }
                    } }
                }
            }
        }
    }

    Entity {
        id("openButton")
        startPosition(200, 150)
        color("22c55e")
        Type { rectangle(400, 130) }
        Events { OnTapped("openButton") { Actions { broadcast("tryCode") } } }
    }
    Entity {
        id("openLabel")
        startPosition(290, 185)
        color("052e16")
        Type { text("TRY", 60) }
        Events { OnTapped { Actions { broadcast("tryCode") } } }
    }
    Entity {
        id("judge")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSignal("tryCode") {
                Actions {
                    delay(0.2) { Actions {
                        whenBar("matches", "==", 3) { Actions {
                            sound("win")
                            broadcast("unlocked")
                            gameOver("LOCK OPENED")
                        } }
                        whenBar("matches", "<", 3) { Actions {
                            sound("buzz")
                            updateBar("lives", -1)
                            setText("message", "Click... wrong code")
                            setVisible("hint", true)
                        } }
                    } }
                }
            }
        }
    }
}
