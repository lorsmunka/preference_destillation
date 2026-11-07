# Bakery Ovens: every loaf bakes at its own pace. Pull it while golden,
# pull burnt ones before they catch fire, and leave the pale dough alone.
Window {
    size(9, 12)
    background("2b1a10")
    music("bg_1")

    Bar("loaves") {
        initial(0)
        OnValue(5) { Actions { setText("board", "10 orders to go") } }
        OnValue(10) { Actions { setText("board", "5 orders to go") } }
    }
    winAt("loaves", 15, "-- ALL ORDERS BAKED --", "win")

    Bar("smoke") {
        initial(0)
        OnValue(1) {
            Actions {
                setOpacity("alarm", 0.6)
                flash("alarm", "ef4444", 0.5)
            }
        }
        OnValue(2) {
            Actions {
                setOpacity("alarm", 1)
                flash("alarm", "ef4444", 0.8)
                sound("beep")
            }
        }
        OnValue(3) {
            Actions {
                sound("lose")
                shake(8, 0.5)
                gameOver("-- THE KITCHEN FILLED WITH SMOKE --")
            }
        }
    }

    Entity {
        id("board")
        startPosition(60, 1080)
        color("fde68a")
        Type { text("15 orders to bake", 44) }
        Events {
            OnElapsedTime {
                Actions { setEnable("loaf3", false) }
            }
            OnElapsedTime(25) {
                Actions {
                    setEnable("loaf3", true)
                    setText("board", "The third oven is lit!")
                    sound("bell")
                }
            }
        }
    }

    Entity {
        id("alarm")
        startPosition(800, 1080)
        color("ef4444")
        Type { circle(30) }
        Events {
            OnElapsedTime {
                Actions { setOpacity(0.15) }
            }
        }
    }

    Entity {
        id("loaf1")
        startPosition(60, 560)
        color("fde68a")
        Type { rectangle(220, 120) }
        var("heat", 0)
        Events {
            OnElapsedTime(random(1.2, 2)) {
                repeatCount(-1)
                Actions {
                    changeVar("heat", 1)
                    whenVar("heat", "==", 2) { Actions { setColor("f59e0b") sound("bell") } }
                    whenVar("heat", "==", 4) { Actions { setColor("44403c") sound("buzz") } }
                    whenVar("heat", "==", 5) {
                        Actions {
                            sound("hurt")
                            shake(6, 0.3)
                            burst(16, "78716c")
                            updateBar("smoke", 1)
                            setColor("fde68a")
                            changeVar("heat", -5)
                        }
                    }
                }
            }
            OnTapped("loaf1") {
                Actions {
                    whenVar("heat", "==", 1) { Actions { sound("thud") flash("ffffff", 0.2) changeVar("heat", -1) } }
                    whenVar("heat", "==", 2) { Actions { sound("coin") burst(12, "f59e0b") updateBar("loaves", 1) setColor("fde68a") changeVar("heat", -2) } }
                    whenVar("heat", "==", 3) { Actions { sound("coin") burst(12, "f59e0b") updateBar("loaves", 1) setColor("fde68a") changeVar("heat", -3) } }
                    whenVar("heat", "==", 4) { Actions { sound("thud") burst(8, "44403c") setColor("fde68a") changeVar("heat", -4) } }
                }
            }
        }
    }

    Entity {
        id("loaf2")
        startPosition(340, 560)
        color("fde68a")
        Type { rectangle(220, 120) }
        var("heat", 0)
        Events {
            OnElapsedTime(random(1, 1.8)) {
                repeatCount(-1)
                Actions {
                    changeVar("heat", 1)
                    whenVar("heat", "==", 2) { Actions { setColor("f59e0b") sound("bell") } }
                    whenVar("heat", "==", 4) { Actions { setColor("44403c") sound("buzz") } }
                    whenVar("heat", "==", 5) {
                        Actions {
                            sound("hurt")
                            shake(6, 0.3)
                            burst(16, "78716c")
                            updateBar("smoke", 1)
                            setColor("fde68a")
                            changeVar("heat", -5)
                        }
                    }
                }
            }
            OnTapped("loaf2") {
                Actions {
                    whenVar("heat", "==", 1) { Actions { sound("thud") flash("ffffff", 0.2) changeVar("heat", -1) } }
                    whenVar("heat", "==", 2) { Actions { sound("coin") burst(12, "f59e0b") updateBar("loaves", 1) setColor("fde68a") changeVar("heat", -2) } }
                    whenVar("heat", "==", 3) { Actions { sound("coin") burst(12, "f59e0b") updateBar("loaves", 1) setColor("fde68a") changeVar("heat", -3) } }
                    whenVar("heat", "==", 4) { Actions { sound("thud") burst(8, "44403c") setColor("fde68a") changeVar("heat", -4) } }
                }
            }
        }
    }

    Entity {
        id("loaf3")
        startPosition(620, 560)
        color("fde68a")
        Type { rectangle(220, 120) }
        var("heat", 0)
        Events {
            OnElapsedTime(random(0.8, 1.5)) {
                repeatCount(-1)
                Actions {
                    changeVar("heat", 1)
                    whenVar("heat", "==", 2) { Actions { setColor("f59e0b") sound("bell") } }
                    whenVar("heat", "==", 4) { Actions { setColor("44403c") sound("buzz") } }
                    whenVar("heat", "==", 5) {
                        Actions {
                            sound("hurt")
                            shake(6, 0.3)
                            burst(16, "78716c")
                            updateBar("smoke", 1)
                            setColor("fde68a")
                            changeVar("heat", -5)
                        }
                    }
                }
            }
            OnTapped("loaf3") {
                Actions {
                    whenVar("heat", "==", 1) { Actions { sound("thud") flash("ffffff", 0.2) changeVar("heat", -1) } }
                    whenVar("heat", "==", 2) { Actions { sound("coin") burst(12, "f59e0b") updateBar("loaves", 1) setColor("fde68a") changeVar("heat", -2) } }
                    whenVar("heat", "==", 3) { Actions { sound("coin") burst(12, "f59e0b") updateBar("loaves", 1) setColor("fde68a") changeVar("heat", -3) } }
                    whenVar("heat", "==", 4) { Actions { sound("thud") burst(8, "44403c") setColor("fde68a") changeVar("heat", -4) } }
                }
            }
        }
    }

    Entity {
        id("hint")
        startPosition(140, 380)
        color("a8a29e")
        Type { text("pull the golden loaves", 40) }
        Events {
            OnElapsedTime(6) {
                Actions { setVisible(false) }
            }
        }
    }
}
