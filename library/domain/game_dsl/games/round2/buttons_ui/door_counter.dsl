# You are the doorman. Let in exactly 12 guests, then open the doors.
Window {
    size(8, 12)
    background("27272a")
    music("bg_3")
    lives(3, "THE CLUB IS CHAOS", "lose")
    Bar("guests") { initial(0) }

    Entity {
        id("sign")
        startPosition(80, 1060)
        color("fafafa")
        Type { text("Tonight: exactly 12 guests", 44) }
    }
    Entity {
        id("info")
        startPosition(80, 950)
        color("a1a1aa")
        Type { text("Use the buttons to count them in", 34) }
    }
    Entity {
        id("plusOne")
        startPosition(80, 640)
        color("4ade80")
        Type { rectangle(200, 160) }
        Events {
            OnTapped("plusOne") {
                Actions {
                    sound("click")
                    updateBar("guests", 1)
                    setText("info", "One more guest")
                }
            }
        }
    }
    Entity {
        id("plusThree")
        startPosition(300, 640)
        color("22d3ee")
        Type { rectangle(200, 160) }
        Events {
            OnTapped("plusThree") {
                Actions {
                    sound("click")
                    updateBar("guests", 3)
                    setText("info", "A group of three arrives")
                }
            }
        }
    }
    Entity {
        id("minusOne")
        startPosition(520, 640)
        color("f87171")
        Type { rectangle(200, 160) }
        Events {
            OnTapped("minusOne") {
                Actions {
                    sound("pop")
                    whenBar("guests", ">", 0) { Actions {
                        updateBar("guests", -1)
                        setText("info", "Someone left")
                    } }
                }
            }
        }
    }
    Entity {
        id("plusLabel")
        startPosition(130, 700)
        color("14532d")
        Type { text("+1", 64) }
        Events { OnTapped { Actions { updateBar("guests", 1) } } }
    }
    Entity {
        id("threeLabel")
        startPosition(350, 700)
        color("164e63")
        Type { text("+3", 64) }
        Events { OnTapped { Actions { updateBar("guests", 3) } } }
    }
    Entity {
        id("minusLabel")
        startPosition(580, 700)
        color("7f1d1d")
        Type { text("-1", 64) }
        Events { OnTapped { Actions { whenBar("guests", ">", 0) { Actions { updateBar("guests", -1) } } } } }
    }
    Entity {
        id("openButton")
        startPosition(180, 280)
        color("fbbf24")
        Type { rectangle(440, 170) }
        Events { OnTapped("openButton") { Actions { broadcast("openDoors") } } }
    }
    Entity {
        id("openLabel")
        startPosition(230, 335)
        color("451a03")
        Type { text("OPEN DOORS", 56) }
        Events { OnTapped { Actions { broadcast("openDoors") } } }
    }
    Entity {
        id("checker")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSignal("openDoors") {
                Actions {
                    whenBar("guests", "==", 12) { Actions {
                        sound("win")
                        gameOver("PERFECT PARTY")
                    } }
                    whenBar("guests", "<", 12) { Actions {
                        sound("buzz")
                        updateBar("lives", -1)
                        setText("info", "Too empty, the crowd boos")
                    } }
                    whenBar("guests", ">", 12) { Actions {
                        sound("buzz")
                        updateBar("lives", -1)
                        setText("info", "Too crowded, the fire marshal frowns")
                    } }
                }
            }
        }
    }
}
