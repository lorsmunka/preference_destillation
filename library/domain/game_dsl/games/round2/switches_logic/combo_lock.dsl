Window {
    size(8, 10)
    background("1c1917")
    music("bg_3")
    Bar("right") {
        initial(0)
        OnValue(3) { Actions { broadcast("opened") } }
    }
    Bar("taps") {
        initial(18)
        OnValue(0) { Actions { gameOver("-- ALARM TRIGGERED --") } }
    }

    Entity {
        id("dial1")
        startPosition(90, 520)
        color("64748b")
        Type { rectangle(170, 170) }
        var("pos", 0)
        Events {
            OnTapped("dial1") {
                Actions {
                    sound("click")
                    updateBar("taps", -1)
                    whenVar("pos", "==", 2) { Actions { updateBar("right", -1) } }
                    changeVar("pos", 1)
                    whenVar("pos", ">", 3) { Actions { changeVar("pos", -4) } }
                    setColor("64748b")
                    whenVar("pos", "==", 0) { Actions { setText("readout1", "0") } }
                    whenVar("pos", "==", 1) { Actions { setText("readout1", "1") } }
                    whenVar("pos", "==", 2) { Actions { setText("readout1", "2") setColor("4ade80") updateBar("right", 1) } }
                    whenVar("pos", "==", 3) { Actions { setText("readout1", "3") } }
                }
            }
        }
    }
    Entity {
        id("dial2")
        startPosition(310, 520)
        color("64748b")
        Type { rectangle(170, 170) }
        var("pos", 0)
        Events {
            OnTapped("dial2") {
                Actions {
                    sound("click")
                    updateBar("taps", -1)
                    whenVar("pos", "==", 1) { Actions { updateBar("right", -1) } }
                    changeVar("pos", 1)
                    whenVar("pos", ">", 3) { Actions { changeVar("pos", -4) } }
                    setColor("64748b")
                    whenVar("pos", "==", 0) { Actions { setText("readout2", "0") } }
                    whenVar("pos", "==", 1) { Actions { setText("readout2", "1") setColor("4ade80") updateBar("right", 1) } }
                    whenVar("pos", "==", 2) { Actions { setText("readout2", "2") } }
                    whenVar("pos", "==", 3) { Actions { setText("readout2", "3") } }
                }
            }
        }
    }
    Entity {
        id("dial3")
        startPosition(530, 520)
        color("64748b")
        Type { rectangle(170, 170) }
        var("pos", 0)
        Events {
            OnTapped("dial3") {
                Actions {
                    sound("click")
                    updateBar("taps", -1)
                    whenVar("pos", "==", 3) { Actions { updateBar("right", -1) } }
                    changeVar("pos", 1)
                    whenVar("pos", ">", 3) { Actions { changeVar("pos", -4) } }
                    setColor("64748b")
                    whenVar("pos", "==", 0) { Actions { setText("readout3", "0") } }
                    whenVar("pos", "==", 1) { Actions { setText("readout3", "1") } }
                    whenVar("pos", "==", 2) { Actions { setText("readout3", "2") } }
                    whenVar("pos", "==", 3) { Actions { setText("readout3", "3") setColor("4ade80") updateBar("right", 1) } }
                }
            }
        }
    }

    Entity { id("readout1") startPosition(150, 560) color("fef3c7") Type { text("0", 90) } }
    Entity { id("readout2") startPosition(370, 560) color("fef3c7") Type { text("0", 90) } }
    Entity { id("readout3") startPosition(590, 560) color("fef3c7") Type { text("0", 90) } }

    Entity {
        id("hint")
        startPosition(90, 800)
        color("a8a29e")
        Type { text("dials: 2 - 1 - 3", 44) }
    }

    Entity {
        id("safe")
        startPosition(200, 150)
        color("facc15")
        Type { diamond(100, 100) }
        Events {
            OnElapsedTime { Actions { setVisible(false) } }
            OnSignal("opened") {
                Actions {
                    setVisible(true)
                    sound("win")
                    burst(24, "facc15")
                    delay(1.2) { Actions { gameOver("-- VAULT OPEN --") } }
                }
            }
        }
    }
    Entity {
        id("vaultDoor")
        startPosition(100, 90)
        color("57534e")
        Type { rectangle(600, 220) }
        door("opened", "clang")
    }
}
