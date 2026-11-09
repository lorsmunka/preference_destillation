Window {
    size(9, 14)
    background("0b132b")
    music("bg_1")
    Bar("scrap") { initial(0) }
    Bar("helpers") { initial(0) }

    Entity {
        id("title")
        startPosition(60, 1300)
        color("5bc0be")
        Type { text("Robot Workshop", 60) }
    }
    Entity {
        id("goal")
        startPosition(60, 1220)
        color("94a3b8")
        Type { text("Collect 150 scrap. Helpers work for you!", 32) }
    }
    Entity {
        id("status")
        startPosition(60, 1150)
        color("e2e8f0")
        Type { text("Tap the pile to gather scrap", 36) }
    }

    Entity {
        id("pile")
        startPosition(250, 700)
        color("64748b")
        Type { polygon(6, 200) }
        Events {
            OnTapped("pile") {
                Actions {
                    sound("clang")
                    updateBar("scrap", 1)
                    burst(4, "cbd5e1")
                    broadcast("checkWin")
                }
            }
        }
    }

    Entity {
        id("helperButton")
        startPosition(150, 260)
        color("3a506b")
        Type { rectangle(600, 150) }
        Events { OnTapped("helperButton") { Actions { broadcast("buyHelper") } } }
    }
    Entity {
        id("helperLabel")
        startPosition(200, 310)
        color("e2e8f0")
        Type { text("BUY HELPER (15 scrap)", 48) }
        Events { OnTapped { Actions { broadcast("buyHelper") } } }
    }

    Entity {
        id("helperBot")
        template(true)
        startPosition(0, 0)
        color("5bc0be")
        Type { rectangle(60, 60) }
        Events {
            OnElapsedTime(1.5) {
                repeatCount(-1)
                Actions {
                    updateBar("scrap", 1)
                    flash("ffffff", 0.15)
                    broadcast("checkWin")
                }
            }
        }
    }

    Entity {
        id("manager")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSignal("buyHelper") {
                Actions {
                    whenBar("helpers", ">=", 4) { Actions { setText("status", "The workshop is full") } }
                    whenBar("helpers", "<", 4) { Actions {
                        whenBar("scrap", "<", 15) { Actions { sound("buzz") setText("status", "Not enough scrap yet") } }
                        whenBar("scrap", ">=", 15) { Actions {
                            sound("pickup")
                            updateBar("scrap", -15)
                            whenBar("helpers", "==", 0) { Actions { spawn("helperBot", 120, 520) } }
                            whenBar("helpers", "==", 1) { Actions { spawn("helperBot", 220, 480) } }
                            whenBar("helpers", "==", 2) { Actions { spawn("helperBot", 600, 520) } }
                            whenBar("helpers", "==", 3) { Actions { spawn("helperBot", 700, 480) } }
                            updateBar("helpers", 1)
                            setText("status", "A new helper joins in")
                        } }
                    } }
                }
            }
            OnSignal("checkWin") {
                Actions {
                    whenBar("scrap", ">=", 150) { Actions { sound("win") gameOver("WORKSHOP COMPLETE") } }
                }
            }
        }
    }
}
