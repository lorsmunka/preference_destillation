Window {
    size(8, 12)
    background("0c1222")
    music("bg_2")
    gravity(0)
    Bar("moves") {
        initial(70)
        OnValue(0) { Actions { gameOver("-- OUT OF MOVES --") } }
    }

    # walls of the three rooms (one cell = 100 px)
    Entity { id("wallA1") startPosition(60, 640) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wallA2") startPosition(160, 640) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wallA3") startPosition(260, 640) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wallA4") startPosition(460, 640) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wallA5") startPosition(560, 640) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wallA6") startPosition(660, 640) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wallB1") startPosition(60, 840) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wallB2") startPosition(260, 840) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wallB3") startPosition(360, 840) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wallB4") startPosition(460, 840) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wallB5") startPosition(560, 840) color("1f2937") solid(true) Type { rectangle(100, 100) } }
    Entity { id("wallB6") startPosition(660, 840) color("1f2937") solid(true) Type { rectangle(100, 100) } }

    Entity {
        id("redGate")
        startPosition(360, 640)
        color("dc2626")
        solid(true)
        Type { rectangle(100, 100) }
        var("open", 0)
        Events {
            OnSignal("flipRed") {
                Actions {
                    whenVar("open", "==", 0) { Actions { delay(0.02) { Actions {
                        changeVar("open", 1)
                        setEnable(false)
                    } } } }
                    whenVar("open", "==", 1) { Actions { delay(0.02) { Actions {
                        changeVar("open", -1)
                        setEnable(true)
                    } } } }
                }
            }
        }
    }
    Entity {
        id("blueGate")
        startPosition(160, 840)
        color("2563eb")
        solid(true)
        Type { rectangle(100, 100) }
        var("open", 0)
        Events {
            OnSignal("flipBlue") {
                Actions {
                    whenVar("open", "==", 0) { Actions { delay(0.02) { Actions {
                        changeVar("open", 1)
                        setEnable(false)
                    } } } }
                    whenVar("open", "==", 1) { Actions { delay(0.02) { Actions {
                        changeVar("open", -1)
                        setEnable(true)
                    } } } }
                }
            }
        }
    }

    Entity {
        id("plateRed")
        startPosition(560, 540)
        color("991b1b")
        Type { rectangle(100, 100) }
        Events { OnCollision("hero") { Actions { sound("click") broadcast("flipRed") } } }
    }
    Entity {
        id("plateBlue")
        startPosition(660, 740)
        color("1e40af")
        Type { rectangle(100, 100) }
        Events { OnCollision("hero") { Actions { sound("click") broadcast("flipBlue") } } }
    }
    Entity {
        id("guard")
        startPosition(60, 740)
        color("f43f5e")
        Type { circle(34) }
        hostile("hero", "-- CAUGHT --", "lose")
        Movement { speed(2.2) Path { waypoint(60, 740) waypoint(460, 740) dwell(0.6) } }
    }

    goal("exit", 360, 1040, 100, 100, "-- ALL ROOMS OPENED --")

    Entity {
        id("hero")
        startPosition(60, 440)
        color("22d3ee")
        Type { circle(40) }
        Movement { GridMove(100, 0.3) { direction(0, 0) } }
        Events {
            OnGridBlocked { Actions { sound("bump") updateBar("moves", -1) } }
        }
    }

    Entity {
        id("padUp")
        startPosition(340, 250)
        color("94a3b8")
        Type { rectangle(120, 120) }
        Events { OnTapped("padUp") { Actions { gridStep("hero", 0, 1) updateBar("moves", -1) } } }
    }
    Entity {
        id("padDown")
        startPosition(340, 70)
        color("94a3b8")
        Type { rectangle(120, 120) }
        Events { OnTapped("padDown") { Actions { gridStep("hero", 0, -1) updateBar("moves", -1) } } }
    }
    Entity {
        id("padLeft")
        startPosition(180, 160)
        color("94a3b8")
        Type { rectangle(120, 120) }
        Events { OnTapped("padLeft") { Actions { gridStep("hero", -1, 0) updateBar("moves", -1) } } }
    }
    Entity {
        id("padRight")
        startPosition(500, 160)
        color("94a3b8")
        Type { rectangle(120, 120) }
        Events { OnTapped("padRight") { Actions { gridStep("hero", 1, 0) updateBar("moves", -1) } } }
    }
}
