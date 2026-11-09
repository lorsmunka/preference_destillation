Window {
    size(8, 10)
    background("111827")
    music("bg_ambient")
    lives(3, "-- THE LIGHTHOUSE WENT DARK --", "lose")
    Bar("step") {
        initial(0)
        OnValue(4) { Actions { sound("win") gameOver("-- THE LIGHT IS ON --") } }
    }

    Entity {
        id("keeper")
        startPosition(380, 900)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(1.0) { Actions { broadcast("replay") } }
            OnSignal("replay") {
                Actions {
                    broadcast("glow3")
                    delay(0.8) { Actions { broadcast("glow1") } }
                    delay(1.6) { Actions { broadcast("glow4") } }
                    delay(2.4) { Actions { broadcast("glow2") } }
                }
            }
        }
    }
    Entity { id("title") startPosition(120, 880) color("94a3b8") Type { text("watch the lamps, then repeat", 40) } }

    Entity {
        id("lamp1")
        startPosition(120, 520)
        color("f87171")
        Type { circle(90) }
        Events {
            OnTapped("lamp1") {
                Actions {
                    flash("ffffff", 0.2)
                    whenBar("step", "==", 1) {
                        Actions {
                            sound("pickup")
                            delay(0.02) { Actions { updateBar("step", 1) } }
                        }
                    }
                    whenBar("step", "!=", 1) {
                        Actions {
                            sound("buzz")
                            updateBar("lives", -1)
                            whenBar("step", "==", 1) { Actions { updateBar("step", -1) } }
                            whenBar("step", "==", 2) { Actions { updateBar("step", -2) } }
                            whenBar("step", "==", 3) { Actions { updateBar("step", -3) } }
                            delay(0.8) { Actions { broadcast("replay") } }
                        }
                    }
                }
            }
            OnSignal("glow1") { Actions { flash("ffffff", 0.45) sound("bell") burst(8, "f87171") } }
        }
    }
    Entity {
        id("lamp2")
        startPosition(480, 520)
        color("60a5fa")
        Type { circle(90) }
        Events {
            OnTapped("lamp2") {
                Actions {
                    flash("ffffff", 0.2)
                    whenBar("step", "==", 3) {
                        Actions {
                            sound("pickup")
                            delay(0.02) { Actions { updateBar("step", 1) } }
                        }
                    }
                    whenBar("step", "!=", 3) {
                        Actions {
                            sound("buzz")
                            updateBar("lives", -1)
                            whenBar("step", "==", 1) { Actions { updateBar("step", -1) } }
                            whenBar("step", "==", 2) { Actions { updateBar("step", -2) } }
                            whenBar("step", "==", 3) { Actions { updateBar("step", -3) } }
                            delay(0.8) { Actions { broadcast("replay") } }
                        }
                    }
                }
            }
            OnSignal("glow2") { Actions { flash("ffffff", 0.45) sound("blip") burst(8, "60a5fa") } }
        }
    }
    Entity {
        id("lamp3")
        startPosition(120, 200)
        color("4ade80")
        Type { circle(90) }
        Events {
            OnTapped("lamp3") {
                Actions {
                    flash("ffffff", 0.2)
                    whenBar("step", "==", 0) {
                        Actions {
                            sound("pickup")
                            delay(0.02) { Actions { updateBar("step", 1) } }
                        }
                    }
                    whenBar("step", "!=", 0) {
                        Actions {
                            sound("buzz")
                            updateBar("lives", -1)
                            whenBar("step", "==", 1) { Actions { updateBar("step", -1) } }
                            whenBar("step", "==", 2) { Actions { updateBar("step", -2) } }
                            whenBar("step", "==", 3) { Actions { updateBar("step", -3) } }
                            delay(0.8) { Actions { broadcast("replay") } }
                        }
                    }
                }
            }
            OnSignal("glow3") { Actions { flash("ffffff", 0.45) sound("bubble") burst(8, "4ade80") } }
        }
    }
    Entity {
        id("lamp4")
        startPosition(480, 200)
        color("facc15")
        Type { circle(90) }
        Events {
            OnTapped("lamp4") {
                Actions {
                    flash("ffffff", 0.2)
                    whenBar("step", "==", 2) {
                        Actions {
                            sound("pickup")
                            delay(0.02) { Actions { updateBar("step", 1) } }
                        }
                    }
                    whenBar("step", "!=", 2) {
                        Actions {
                            sound("buzz")
                            updateBar("lives", -1)
                            whenBar("step", "==", 1) { Actions { updateBar("step", -1) } }
                            whenBar("step", "==", 2) { Actions { updateBar("step", -2) } }
                            whenBar("step", "==", 3) { Actions { updateBar("step", -3) } }
                            delay(0.8) { Actions { broadcast("replay") } }
                        }
                    }
                }
            }
            OnSignal("glow4") { Actions { flash("ffffff", 0.45) sound("glass") burst(8, "facc15") } }
        }
    }
}
