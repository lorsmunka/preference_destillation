Window {
    size(8, 12)
    background("0f1b14")
    music("bg_nature")
    lives(3, "-- TOO MANY WRONG TURNS --", "lose")
    winAt("score", 6, "-- ALL DELIVERED --", "win")

    # track
    Entity { id("railMain") startPosition(40, 770) color("3f3f46") Type { rectangle(700, 40) } }
    Entity { id("railUp") startPosition(380, 770) color("3f3f46") Type { rectangle(40, 300) } }

    Entity {
        id("junction")
        startPosition(360, 740)
        color("a3e635")
        Type { rectangle(100, 100) }
        var("branch", 0)
        Events {
            OnSignal("flip") {
                Actions {
                    whenVar("branch", "==", 0) { Actions { delay(0.02) { Actions {
                        changeVar("branch", 1)
                        setColor("f472b6")
                    } } } }
                    whenVar("branch", "==", 1) { Actions { delay(0.02) { Actions {
                        changeVar("branch", -1)
                        setColor("a3e635")
                    } } } }
                }
            }
            OnCollision("redTrain") {
                Actions {
                    whenVar("branch", "==", 1) { Actions { setDirection("redTrain", 0, 1) } }
                    whenVar("branch", "==", 0) { Actions { setDirection("redTrain", 1, 0) } }
                }
            }
            OnCollision("blueTrain") {
                Actions {
                    whenVar("branch", "==", 1) { Actions { setDirection("blueTrain", 0, 1) } }
                    whenVar("branch", "==", 0) { Actions { setDirection("blueTrain", 1, 0) } }
                }
            }
        }
    }

    Entity {
        id("depotTop")
        startPosition(360, 1040)
        color("991b1b")
        Type { rectangle(100, 100) }
        Events {
            OnCollision("redTrain") {
                Actions {
                    sound("coin")
                    updateBar("score", 1)
                    setEnable("redTrain", false)
                    delay(1.0) { Actions { broadcast("sendBlue") } }
                }
            }
            OnCollision("blueTrain") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    setEnable("blueTrain", false)
                    delay(1.0) { Actions { broadcast("sendRed") } }
                }
            }
        }
    }
    Entity {
        id("depotRight")
        startPosition(660, 740)
        color("1e3a8a")
        Type { rectangle(100, 100) }
        Events {
            OnCollision("blueTrain") {
                Actions {
                    sound("coin")
                    updateBar("score", 1)
                    setEnable("blueTrain", false)
                    delay(1.0) { Actions { broadcast("sendRed") } }
                }
            }
            OnCollision("redTrain") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    setEnable("redTrain", false)
                    delay(1.0) { Actions { broadcast("sendBlue") } }
                }
            }
        }
    }

    Entity {
        id("redTrain")
        startPosition(60, 740)
        color("ef4444")
        Type { circle(40) }
        Movement { GridMove(100, 0.6) { direction(1, 0) } }
        Events {
            OnSignal("sendRed") {
                Actions {
                    setEnable(true)
                    setPosition("redTrain", 60, 740, 0)
                    setDirection("redTrain", 1, 0)
                }
            }
            OnGridBlocked { Actions { gameOver("-- DERAILED --") } }
        }
    }
    Entity {
        id("blueTrain")
        startPosition(60, 740)
        color("3b82f6")
        Type { circle(40) }
        Movement { GridMove(100, 0.6) { direction(1, 0) } }
        Events {
            OnElapsedTime { Actions { setEnable(false) } }
            OnSignal("sendBlue") {
                Actions {
                    setEnable(true)
                    setPosition("blueTrain", 60, 740, 0)
                    setDirection("blueTrain", 1, 0)
                }
            }
            OnGridBlocked { Actions { gameOver("-- DERAILED --") } }
        }
    }

    Entity {
        id("switchBtn")
        startPosition(200, 200)
        color("f472b6")
        Type { rectangle(400, 220) }
        Events {
            OnTapped("switchBtn") { Actions { sound("click") broadcast("flip") } }
        }
    }
    Entity { id("btnLabel") startPosition(250, 290) color("1f2937") Type { text("FLIP THE SWITCH", 46) } }
}
