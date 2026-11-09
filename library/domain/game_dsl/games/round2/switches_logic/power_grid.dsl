Window {
    size(8, 10)
    background("0b1120")
    music("bg_5")
    Bar("lit") {
        initial(4)
        OnValue(0) { Actions { broadcast("solved") } }
    }
    Bar("moves") {
        initial(12)
        OnValue(0) { Actions { gameOver("-- THE GRID SHORTED OUT --") } }
    }

    Entity {
        id("banner")
        startPosition(150, 880)
        color("94a3b8")
        Type { text("switch every light off", 48) }
        Events {
            OnSignal("solved") {
                Actions {
                    setText("banner", "all dark!")
                    sound("win")
                    delay(1.0) { Actions { gameOver("-- GRID POWERED DOWN --") } }
                }
            }
        }
    }

    Entity {
        id("cellA3")
        startPosition(110, 700)
        color("fde047")
        Type { rectangle(180, 180) }
        var("lit", 1)
        Events {
            OnTapped("cellA3") {
                Actions {
                    sound("click")
                    updateBar("moves", -1)
                    broadcast("flipA3")
                    broadcast("flipB3")
                    broadcast("flipA2")
                }
            }
            OnSignal("flipA3") {
                Actions {
                    whenVar("lit", "==", 0) { Actions { delay(0.02) { Actions {
                        changeVar("lit", 1)
                        setColor("fde047")
                        updateBar("lit", 1)
                    } } } }
                    whenVar("lit", "==", 1) { Actions { delay(0.05) { Actions {
                        changeVar("lit", -1)
                        setColor("334155")
                        updateBar("lit", -1)
                    } } } }
                }
            }
        }
    }
    Entity {
        id("cellB3")
        startPosition(310, 700)
        color("334155")
        Type { rectangle(180, 180) }
        var("lit", 0)
        Events {
            OnTapped("cellB3") {
                Actions {
                    sound("click")
                    updateBar("moves", -1)
                    broadcast("flipB3")
                    broadcast("flipA3")
                    broadcast("flipC3")
                    broadcast("flipB2")
                }
            }
            OnSignal("flipB3") {
                Actions {
                    whenVar("lit", "==", 0) { Actions { delay(0.02) { Actions {
                        changeVar("lit", 1)
                        setColor("fde047")
                        updateBar("lit", 1)
                    } } } }
                    whenVar("lit", "==", 1) { Actions { delay(0.05) { Actions {
                        changeVar("lit", -1)
                        setColor("334155")
                        updateBar("lit", -1)
                    } } } }
                }
            }
        }
    }
    Entity {
        id("cellC3")
        startPosition(510, 700)
        color("334155")
        Type { rectangle(180, 180) }
        var("lit", 0)
        Events {
            OnTapped("cellC3") {
                Actions {
                    sound("click")
                    updateBar("moves", -1)
                    broadcast("flipC3")
                    broadcast("flipB3")
                    broadcast("flipC2")
                }
            }
            OnSignal("flipC3") {
                Actions {
                    whenVar("lit", "==", 0) { Actions { delay(0.02) { Actions {
                        changeVar("lit", 1)
                        setColor("fde047")
                        updateBar("lit", 1)
                    } } } }
                    whenVar("lit", "==", 1) { Actions { delay(0.05) { Actions {
                        changeVar("lit", -1)
                        setColor("334155")
                        updateBar("lit", -1)
                    } } } }
                }
            }
        }
    }
    Entity {
        id("cellA2")
        startPosition(110, 500)
        color("334155")
        Type { rectangle(180, 180) }
        var("lit", 0)
        Events {
            OnTapped("cellA2") {
                Actions {
                    sound("click")
                    updateBar("moves", -1)
                    broadcast("flipA2")
                    broadcast("flipB2")
                    broadcast("flipA1")
                    broadcast("flipA3")
                }
            }
            OnSignal("flipA2") {
                Actions {
                    whenVar("lit", "==", 0) { Actions { delay(0.02) { Actions {
                        changeVar("lit", 1)
                        setColor("fde047")
                        updateBar("lit", 1)
                    } } } }
                    whenVar("lit", "==", 1) { Actions { delay(0.05) { Actions {
                        changeVar("lit", -1)
                        setColor("334155")
                        updateBar("lit", -1)
                    } } } }
                }
            }
        }
    }
    Entity {
        id("cellB2")
        startPosition(310, 500)
        color("fde047")
        Type { rectangle(180, 180) }
        var("lit", 1)
        Events {
            OnTapped("cellB2") {
                Actions {
                    sound("click")
                    updateBar("moves", -1)
                    broadcast("flipB2")
                    broadcast("flipA2")
                    broadcast("flipC2")
                    broadcast("flipB1")
                    broadcast("flipB3")
                }
            }
            OnSignal("flipB2") {
                Actions {
                    whenVar("lit", "==", 0) { Actions { delay(0.02) { Actions {
                        changeVar("lit", 1)
                        setColor("fde047")
                        updateBar("lit", 1)
                    } } } }
                    whenVar("lit", "==", 1) { Actions { delay(0.05) { Actions {
                        changeVar("lit", -1)
                        setColor("334155")
                        updateBar("lit", -1)
                    } } } }
                }
            }
        }
    }
    Entity {
        id("cellC2")
        startPosition(510, 500)
        color("fde047")
        Type { rectangle(180, 180) }
        var("lit", 1)
        Events {
            OnTapped("cellC2") {
                Actions {
                    sound("click")
                    updateBar("moves", -1)
                    broadcast("flipC2")
                    broadcast("flipB2")
                    broadcast("flipC1")
                    broadcast("flipC3")
                }
            }
            OnSignal("flipC2") {
                Actions {
                    whenVar("lit", "==", 0) { Actions { delay(0.02) { Actions {
                        changeVar("lit", 1)
                        setColor("fde047")
                        updateBar("lit", 1)
                    } } } }
                    whenVar("lit", "==", 1) { Actions { delay(0.05) { Actions {
                        changeVar("lit", -1)
                        setColor("334155")
                        updateBar("lit", -1)
                    } } } }
                }
            }
        }
    }
    Entity {
        id("cellA1")
        startPosition(110, 300)
        color("334155")
        Type { rectangle(180, 180) }
        var("lit", 0)
        Events {
            OnTapped("cellA1") {
                Actions {
                    sound("click")
                    updateBar("moves", -1)
                    broadcast("flipA1")
                    broadcast("flipB1")
                    broadcast("flipA2")
                }
            }
            OnSignal("flipA1") {
                Actions {
                    whenVar("lit", "==", 0) { Actions { delay(0.02) { Actions {
                        changeVar("lit", 1)
                        setColor("fde047")
                        updateBar("lit", 1)
                    } } } }
                    whenVar("lit", "==", 1) { Actions { delay(0.05) { Actions {
                        changeVar("lit", -1)
                        setColor("334155")
                        updateBar("lit", -1)
                    } } } }
                }
            }
        }
    }
    Entity {
        id("cellB1")
        startPosition(310, 300)
        color("fde047")
        Type { rectangle(180, 180) }
        var("lit", 1)
        Events {
            OnTapped("cellB1") {
                Actions {
                    sound("click")
                    updateBar("moves", -1)
                    broadcast("flipB1")
                    broadcast("flipA1")
                    broadcast("flipC1")
                    broadcast("flipB2")
                }
            }
            OnSignal("flipB1") {
                Actions {
                    whenVar("lit", "==", 0) { Actions { delay(0.02) { Actions {
                        changeVar("lit", 1)
                        setColor("fde047")
                        updateBar("lit", 1)
                    } } } }
                    whenVar("lit", "==", 1) { Actions { delay(0.05) { Actions {
                        changeVar("lit", -1)
                        setColor("334155")
                        updateBar("lit", -1)
                    } } } }
                }
            }
        }
    }
    Entity {
        id("cellC1")
        startPosition(510, 300)
        color("334155")
        Type { rectangle(180, 180) }
        var("lit", 0)
        Events {
            OnTapped("cellC1") {
                Actions {
                    sound("click")
                    updateBar("moves", -1)
                    broadcast("flipC1")
                    broadcast("flipB1")
                    broadcast("flipC2")
                }
            }
            OnSignal("flipC1") {
                Actions {
                    whenVar("lit", "==", 0) { Actions { delay(0.02) { Actions {
                        changeVar("lit", 1)
                        setColor("fde047")
                        updateBar("lit", 1)
                    } } } }
                    whenVar("lit", "==", 1) { Actions { delay(0.05) { Actions {
                        changeVar("lit", -1)
                        setColor("334155")
                        updateBar("lit", -1)
                    } } } }
                }
            }
        }
    }
}
