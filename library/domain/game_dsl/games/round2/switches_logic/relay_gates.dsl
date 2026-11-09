Window {
    size(8, 10)
    background("0b132b")
    music("bg_1")
    Bar("power") {
        initial(0)
        OnValue(3) { Actions { broadcast("lit") } }
    }

    Entity {
        id("generator")
        startPosition(80, 80)
        color("facc15")
        Type { circle(70) }
        var("charge", 0)
        Events {
            OnTapped("generator") {
                Actions {
                    sound("pop")
                    changeVar("charge", 1)
                    burst(5, "facc15")
                    whenVar("charge", ">=", 6) {
                        Actions {
                            changeVar("charge", -6)
                            broadcast("power1")
                        }
                    }
                }
            }
            OnElapsedTime(1.0) {
                repeatCount(-1)
                Actions { whenVar("charge", ">", 0) { Actions { changeVar("charge", -1) } } }
            }
            OnElapsedTime(30.0) { Actions { gameOver("-- BLACKOUT --") } }
        }
    }

    Entity {
        id("relay1")
        startPosition(500, 220)
        color("334155")
        Type { polygon(6, 50) }
        Events {
            OnSignal("power1") {
                Actions {
                    setColor("38bdf8")
                    sound("blip")
                    delay(0.8) { Actions { updateBar("power", 1) broadcast("power2") } }
                }
            }
        }
    }
    Entity {
        id("relay2")
        startPosition(180, 420)
        color("334155")
        Type { polygon(6, 50) }
        Events {
            OnSignal("power2") {
                Actions {
                    setColor("38bdf8")
                    sound("blip")
                    delay(0.8) { Actions { updateBar("power", 1) broadcast("power3") } }
                }
            }
        }
    }
    Entity {
        id("relay3")
        startPosition(500, 620)
        color("334155")
        Type { polygon(6, 50) }
        Events {
            OnSignal("power3") {
                Actions {
                    setColor("38bdf8")
                    sound("blip")
                    delay(0.8) { Actions { updateBar("power", 1) } }
                }
            }
        }
    }

    Entity {
        id("shutter")
        startPosition(100, 820)
        color("7c3aed")
        Type { rectangle(600, 30) }
        door("lit", "clang")
    }
    Entity {
        id("beacon")
        startPosition(330, 890)
        color("f97316")
        Type { star(5, 36, 16) }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("lit") {
                Actions {
                    setEnable(true)
                    burst(20, "f97316")
                    delay(1.0) { Actions { sound("win") gameOver("-- BEACON LIT --") } }
                }
            }
        }
    }
}
