# Reactor Shift: run a power station for one shift and deliver 150 units to the city.
# Swipe up to pull the control rods out (more power, more heat), swipe down to push them in (cooler, slower).
# Tap the pump to spend coolant on the core. Above heat 14 the safety system counts down to an automatic
# scram; at 18 the core melts down. If the shift ends before the quota is met, the plant is shut down.
Window {
    size(9, 14)
    background("0b1210")
    music("bg_4")
    gravity(0)

    Bar("heat") {
        initial(4)
        OnValue(9) { Actions { broadcast("normal") } }
        OnValue(10) { Actions { broadcast("warn") } }
        OnValue(13) { Actions { broadcast("warn") } }
        OnValue(14) { Actions { broadcast("critical") } }
        OnValue(18) {
            Actions {
                sound("death")
                shake(14, 1.2)
                burst("core", 60, "f97316")
                gameOver("-- MELTDOWN --")
            }
        }
    }

    Bar("rods") {
        initial(2)
        OnValue(0) { Actions { setPosition("rodBank", 350, 880, 0.4) } }
        OnValue(1) { Actions { setPosition("rodBank", 350, 820, 0.4) } }
        OnValue(2) { Actions { setPosition("rodBank", 350, 760, 0.4) } }
        OnValue(3) { Actions { setPosition("rodBank", 350, 700, 0.4) } }
        OnValue(4) { Actions { setPosition("rodBank", 350, 640, 0.4) } }
    }

    Bar("output") {
        initial(0)
        OnValue(50) { Actions { broadcast("district1") } }
        OnValue(100) { Actions { broadcast("district2") } }
    }
    winAt("output", 150, "-- THE CITY IS POWERED --", "win")

    Bar("coolant") { initial(6) }

    Bar("shift") {
        initial(0)
        OnValue(75) { Actions { broadcast("lateShift") } }
        OnValue(100) { Actions { sound("lose") gameOver("-- SHIFT OVER, QUOTA MISSED --") } }
    }

    Entity {
        id("status")
        startPosition(40, 1340)
        color("4ade80")
        Type { text("CORE STABLE", 40) }
        Events {
            OnSignal("normal") { Actions { setText("status", "CORE STABLE") setColor("4ade80") } }
            OnSignal("warn") { Actions { setText("status", "CORE HOT") setColor("fbbf24") } }
            OnSignal("critical") {
                Actions {
                    setText("status", "SCRAM IN 3")
                    setColor("ef4444")
                    delay(1) { Actions { whenBar("heat", ">=", 14) { Actions {
                        setText("status", "SCRAM IN 2")
                        delay(1) { Actions { whenBar("heat", ">=", 14) { Actions {
                            setText("status", "SCRAM IN 1")
                            delay(1) { Actions { whenBar("heat", ">=", 14) { Actions {
                                setText("status", "AUTO SCRAM")
                                sound("clang")
                                shake(8, 0.5)
                                loopCount(4)
                                whenBar("rods", "<", 4) { Actions { updateBar("rods", 1) } }
                            } } } }
                        } } } }
                    } } } }
                }
            }
        }
    }

    Entity {
        id("clock")
        startPosition(40, 1290)
        color("94a3b8")
        Type { text("Shift started", 30) }
        Events {
            OnElapsedTime { Actions { ramp("shift", 0, 100, 140) } }
            OnSignal("lateShift") { Actions { setText("clock", "Last quarter of the shift!") setColor("fbbf24") sound("bell") } }
        }
    }

    Entity {
        id("core")
        startPosition(300, 560)
        color("0ea5e9")
        Type { circle(150) }
        Events {
            OnElapsedTime(1) {
                repeatCount(-1)
                Actions {
                    whenBar("rods", "==", 0) { Actions { updateBar("heat", 1) loopCount(3) updateBar("output", 1) } }
                    whenBar("rods", "==", 1) { Actions { loopCount(2) updateBar("output", 1) } }
                    whenBar("rods", "==", 2) { Actions { updateBar("output", 1) } }
                    whenBar("rods", "==", 4) { Actions { whenBar("heat", ">", 0) { Actions { updateBar("heat", -1) } } } }
                }
            }
            OnElapsedTime(2) {
                repeatCount(-1)
                Actions {
                    whenBar("rods", "==", 1) { Actions { updateBar("heat", 1) } }
                    whenBar("rods", "==", 3) { Actions { updateBar("output", 1) whenBar("heat", ">", 0) { Actions { updateBar("heat", -1) } } } }
                }
            }
            OnSignal("normal") { Actions { setColor("0ea5e9") } }
            OnSignal("warn") { Actions { setColor("f59e0b") } }
            OnSignal("critical") { Actions { setColor("dc2626") flash("ffffff", 0.3) } }
        }
    }

    Entity {
        id("rodBank")
        startPosition(350, 760)
        color("cc334155")
        Type { rectangle(200, 260) }
    }

    Entity {
        id("controls")
        startPosition(40, 60)
        color("cbd5e1")
        Type { text("swipe up: rods out   swipe down: rods in   tap pump: cool", 24) }
        Events {
            OnSwipe("up") { Actions { whenBar("rods", ">", 0) { Actions { updateBar("rods", -1) sound("click") } } } }
            OnSwipe("down") { Actions { whenBar("rods", "<", 4) { Actions { updateBar("rods", 1) sound("click") } } } }
        }
    }

    Entity {
        id("pump")
        startPosition(660, 300)
        color("38bdf8")
        Type { rectangle(160, 120) }
        var("jammed", 0)
        Events {
            OnTapped("pump") {
                Actions {
                    whenVar("jammed", "==", 1) { Actions { sound("buzz") flash("ef4444", 0.2) } }
                    whenVar("jammed", "==", 0) { Actions {
                        whenBar("coolant", "==", 0) { Actions { sound("buzz") } }
                        whenBar("coolant", ">", 0) { Actions { whenBar("heat", ">", 0) { Actions {
                            updateBar("coolant", -1)
                            updateBar("heat", -1)
                            sound("bubble")
                            burst(10, "7dd3fc")
                            flash("core", "7dd3fc", 0.3)
                        } } } }
                    } }
                }
            }
            OnSignal("pumpJam") {
                Actions { whenVar("jammed", "==", 0) { Actions {
                    changeVar("jammed", 1)
                    setOpacity(0.3)
                    sound("clang")
                    setText("status", "PUMP JAMMED")
                    delay(5) { Actions { changeVar("jammed", -1) setOpacity(1) sound("blip") } }
                } } }
            }
        }
    }

    Entity {
        id("tank")
        startPosition(80, 300)
        color("1e40af")
        Type { rectangle(140, 200) }
        Events {
            OnElapsedTime(4) { repeatCount(-1) Actions { whenBar("coolant", "<", 6) { Actions { updateBar("coolant", 1) burst(4, "93c5fd") } } } }
        }
    }

    Entity {
        id("tower")
        startPosition(60, 1000)
        color("475569")
        Type { rectangle(140, 200) }
        Events {
            OnElapsedTime(1.2) { repeatCount(-1) Actions { whenBar("rods", "<=", 2) { Actions { spawn("steam", 90, 1200) } } } }
        }
    }

    Entity {
        id("steam")
        template(true)
        color("e2e8f0")
        Type { circle(30) }
        Movement { speed(0.6) AutoMove { direction(0, 1) } }
        Events {
            OnElapsedTime {
                Actions {
                    setScale(0.5)
                    delay(0.6) { Actions { setScale(0.9) setOpacity(0.6)
                        delay(0.6) { Actions { setScale(1.3) setOpacity(0.3)
                            delay(0.6) { Actions { deleteEntity() } } } } } }
                }
            }
        }
    }

    Entity {
        id("siren")
        startPosition(800, 900)
        color("ef4444")
        Type { circle(30) }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("critical") { Actions { setEnable(true) } }
            OnSignal("warn") { Actions { setEnable(false) } }
            OnSignal("normal") { Actions { setEnable(false) } }
            OnElapsedTime(0.8) { repeatCount(-1) Actions { sound("beep") flash("ffffff", 0.3) } }
        }
    }

    Entity {
        id("district1")
        startPosition(560, 1040)
        color("1f2937")
        Type { rectangle(140, 120) }
        Events { OnSignal("district1") { Actions { setColor("fde047") burst(20, "fef08a") sound("sparkle") } } }
    }
    Entity {
        id("district2")
        startPosition(720, 1040)
        color("1f2937")
        Type { rectangle(140, 170) }
        Events { OnSignal("district2") { Actions { setColor("fde047") burst(20, "fef08a") sound("sparkle") } } }
    }

    Entity {
        id("dispatcher")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            Sequence {
                Phase(4, 12) { Actions { randomAction {
                    Actions { broadcast("pumpJam") }
                    Actions { updateBar("heat", 1) flash("core", "fb923c", 0.4) }
                    Actions { }
                } } }
                Phase(-1, 8) { Actions { randomAction {
                    Actions { broadcast("pumpJam") }
                    Actions { updateBar("heat", 1) updateBar("heat", 1) flash("core", "fb923c", 0.4) sound("thud") }
                    Actions { whenBar("coolant", ">", 0) { Actions { updateBar("coolant", -1) } } whenBar("coolant", ">", 0) { Actions { updateBar("coolant", -1) } } burst("tank", 12, "60a5fa") }
                } } }
            }
        }
    }
}
