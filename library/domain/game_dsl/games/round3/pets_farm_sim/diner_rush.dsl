Window {
    size(8, 12)
    background("1c1917")
    music("bg_2")
    lives(3, "THE DINER CLOSED")
    winAt("served", 6, "FIVE STAR SERVICE", "win2")
    Bar("served") { initial(0) }
    Bar("plated") { initial(0) }
    Bar("cook") {
        initial(0)
        OnValue(3) { Actions { broadcast("cooked") } }
    }

    Entity {
        id("customer")
        template(true)
        startPosition(0, 0)
        color("fda4af")
        Type { circle(50) }
        Events {
            OnTapped {
                Actions {
                    whenBar("plated", "==", 2) {
                        Actions {
                            updateBar("plated", -2)
                            updateBar("served", 1)
                            sound("coin")
                            burst(10, "fde047")
                            broadcast("served")
                            deleteEntity()
                        }
                    }
                }
            }
            OnElapsedTime(10) {
                Actions {
                    updateBar("lives", -1)
                    sound("hurt")
                    deleteEntity()
                }
            }
            OnElapsedTime(6) { Actions { setColor("ef4444") } }
        }
    }
    Entity {
        id("door")
        startPosition(0, 1000)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(3) {
                repeatCount(-1)
                Actions {
                    randomAction {
                        Actions { spawn("customer", 120, 800) }
                        Actions { spawn("customer", 350, 800) }
                        Actions { spawn("customer", 580, 800) }
                    }
                }
            }
        }
    }
    Entity {
        id("counter")
        startPosition(40, 600)
        color("57534e")
        Type { rectangle(720, 40) }
    }
    Entity {
        id("oven")
        startPosition(80, 200)
        color("78716c")
        Type { rectangle(220, 220) }
        Events {
            OnTapped("oven") {
                Actions {
                    whenBar("plated", "==", 0) {
                        Actions {
                            updateBar("plated", 1)
                            setColor("ea580c")
                            ramp("cook", 0, 3, 3)
                            sound("click")
                        }
                    }
                }
            }
            OnSignal("cooked") {
                Actions {
                    updateBar("plated", 1)
                    setColor("78716c")
                    sound("bell")
                    broadcast("plateReady")
                }
            }
        }
    }
    Entity {
        id("plate")
        startPosition(400, 260)
        color("475569")
        Type { circle(70) }
        Events {
            OnSignal("plateReady") { Actions { setColor("86efac") setScale(1.3) } }
            OnSignal("served") { Actions { setColor("475569") setScale(1) } }
        }
    }
    Entity {
        id("ovenLabel")
        startPosition(120, 90)
        color("fafaf9")
        Type { text("OVEN", 40) }
    }
    Entity {
        id("hint")
        startPosition(60, 1120)
        color("a8a29e")
        Type { text("Cook in the oven, then tap a customer", 28) }
    }
}
