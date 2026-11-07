Window {
    size(8, 12)
    background("1a1626")
    gravity(0)

    Bar("bet") { initial(0) }

    Entity {
        id("slot1")
        startPosition(50, 20)
        color("ef4444")
        Type { rectangle(100, 110) }
        var("isPicked", 0)
        Events {
            OnTapped("slot1") {
                Actions {
                    whenBar("bet", "==", 0) { Actions {
                        sound("click")
                        changeVar("isPicked", 1)
                        updateBar("bet", 1)
                        setColor("ffffff")
                        flash("ffffff", 0.2)
                        burst(8, "ffffff")
                    } }
                }
            }
            OnSignal("jack1") {
                Actions { whenVar("isPicked", "==", 1) { Actions {
                    sound("win")
                    burst(18, "ffd700")
                    shake(8, 0.5)
                    flash("ffd700", 0.5)
                    gameOver("-- JACKPOT --")
                } } }
            }
            OnSignal("small1") {
                Actions { whenVar("isPicked", "==", 1) { Actions {
                    sound("coin")
                    burst(14, "cbd5e1")
                    shake(4, 0.3)
                    flash("cbd5e1", 0.4)
                    gameOver("-- SMALL WIN --")
                } } }
            }
            OnSignal("bonus1") {
                Actions { whenVar("isPicked", "==", 1) { Actions {
                    sound("pickup")
                    burst(10, "94a3b8")
                    flash("94a3b8", 0.3)
                    gameOver("-- BONUS --")
                } } }
            }
        }
    }
    Entity {
        id("slot2")
        startPosition(200, 20)
        color("f59e0b")
        Type { rectangle(100, 110) }
        var("isPicked", 0)
        Events {
            OnTapped("slot2") {
                Actions {
                    whenBar("bet", "==", 0) { Actions {
                        sound("click")
                        changeVar("isPicked", 1)
                        updateBar("bet", 1)
                        setColor("ffffff")
                        flash("ffffff", 0.2)
                        burst(8, "ffffff")
                    } }
                }
            }
            OnSignal("jack2") {
                Actions { whenVar("isPicked", "==", 1) { Actions {
                    sound("win")
                    burst(18, "ffd700")
                    shake(8, 0.5)
                    flash("ffd700", 0.5)
                    gameOver("-- JACKPOT --")
                } } }
            }
            OnSignal("small2") {
                Actions { whenVar("isPicked", "==", 1) { Actions {
                    sound("coin")
                    burst(14, "cbd5e1")
                    shake(4, 0.3)
                    flash("cbd5e1", 0.4)
                    gameOver("-- SMALL WIN --")
                } } }
            }
            OnSignal("bonus2") {
                Actions { whenVar("isPicked", "==", 1) { Actions {
                    sound("pickup")
                    burst(10, "94a3b8")
                    flash("94a3b8", 0.3)
                    gameOver("-- BONUS --")
                } } }
            }
        }
    }
    Entity {
        id("slot3")
        startPosition(350, 20)
        color("22c55e")
        Type { rectangle(100, 110) }
        var("isPicked", 0)
        Events {
            OnTapped("slot3") {
                Actions {
                    whenBar("bet", "==", 0) { Actions {
                        sound("click")
                        changeVar("isPicked", 1)
                        updateBar("bet", 1)
                        setColor("ffffff")
                        flash("ffffff", 0.2)
                        burst(8, "ffffff")
                    } }
                }
            }
            OnSignal("jack3") {
                Actions { whenVar("isPicked", "==", 1) { Actions {
                    sound("win")
                    burst(18, "ffd700")
                    shake(8, 0.5)
                    flash("ffd700", 0.5)
                    gameOver("-- JACKPOT --")
                } } }
            }
            OnSignal("small3") {
                Actions { whenVar("isPicked", "==", 1) { Actions {
                    sound("coin")
                    burst(14, "cbd5e1")
                    shake(4, 0.3)
                    flash("cbd5e1", 0.4)
                    gameOver("-- SMALL WIN --")
                } } }
            }
            OnSignal("bonus3") {
                Actions { whenVar("isPicked", "==", 1) { Actions {
                    sound("pickup")
                    burst(10, "94a3b8")
                    flash("94a3b8", 0.3)
                    gameOver("-- BONUS --")
                } } }
            }
        }
    }
    Entity {
        id("slot4")
        startPosition(500, 20)
        color("3b82f6")
        Type { rectangle(100, 110) }
        var("isPicked", 0)
        Events {
            OnTapped("slot4") {
                Actions {
                    whenBar("bet", "==", 0) { Actions {
                        sound("click")
                        changeVar("isPicked", 1)
                        updateBar("bet", 1)
                        setColor("ffffff")
                        flash("ffffff", 0.2)
                        burst(8, "ffffff")
                    } }
                }
            }
            OnSignal("jack4") {
                Actions { whenVar("isPicked", "==", 1) { Actions {
                    sound("win")
                    burst(18, "ffd700")
                    shake(8, 0.5)
                    flash("ffd700", 0.5)
                    gameOver("-- JACKPOT --")
                } } }
            }
            OnSignal("small4") {
                Actions { whenVar("isPicked", "==", 1) { Actions {
                    sound("coin")
                    burst(14, "cbd5e1")
                    shake(4, 0.3)
                    flash("cbd5e1", 0.4)
                    gameOver("-- SMALL WIN --")
                } } }
            }
            OnSignal("bonus4") {
                Actions { whenVar("isPicked", "==", 1) { Actions {
                    sound("pickup")
                    burst(10, "94a3b8")
                    flash("94a3b8", 0.3)
                    gameOver("-- BONUS --")
                } } }
            }
        }
    }
    Entity {
        id("slot5")
        startPosition(650, 20)
        color("a855f7")
        Type { rectangle(100, 110) }
        var("isPicked", 0)
        Events {
            OnTapped("slot5") {
                Actions {
                    whenBar("bet", "==", 0) { Actions {
                        sound("click")
                        changeVar("isPicked", 1)
                        updateBar("bet", 1)
                        setColor("ffffff")
                        flash("ffffff", 0.2)
                        burst(8, "ffffff")
                    } }
                }
            }
            OnSignal("jack5") {
                Actions { whenVar("isPicked", "==", 1) { Actions {
                    sound("win")
                    burst(18, "ffd700")
                    shake(8, 0.5)
                    flash("ffd700", 0.5)
                    gameOver("-- JACKPOT --")
                } } }
            }
            OnSignal("small5") {
                Actions { whenVar("isPicked", "==", 1) { Actions {
                    sound("coin")
                    burst(14, "cbd5e1")
                    shake(4, 0.3)
                    flash("cbd5e1", 0.4)
                    gameOver("-- SMALL WIN --")
                } } }
            }
            OnSignal("bonus5") {
                Actions { whenVar("isPicked", "==", 1) { Actions {
                    sound("pickup")
                    burst(10, "94a3b8")
                    flash("94a3b8", 0.3)
                    gameOver("-- BONUS --")
                } } }
            }
        }
    }

    Entity {
        id("jackpotBall")
        template(true)
        startPosition(0, 0)
        color("ffd700")
        Type { star(5, 40, 18) }
        Movement { speed(4.25) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("slot1") { Actions { broadcast("jack1") deleteEntity() } }
            OnCollision("slot2") { Actions { broadcast("jack2") deleteEntity() } }
            OnCollision("slot3") { Actions { broadcast("jack3") deleteEntity() } }
            OnCollision("slot4") { Actions { broadcast("jack4") deleteEntity() } }
            OnCollision("slot5") { Actions { broadcast("jack5") deleteEntity() } }
        }
    }

    Entity {
        id("smallBall")
        template(true)
        startPosition(0, 0)
        color("cbd5e1")
        Type { diamond(80, 80) }
        Movement { speed(4.25) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("slot1") { Actions { broadcast("small1") deleteEntity() } }
            OnCollision("slot2") { Actions { broadcast("small2") deleteEntity() } }
            OnCollision("slot3") { Actions { broadcast("small3") deleteEntity() } }
            OnCollision("slot4") { Actions { broadcast("small4") deleteEntity() } }
            OnCollision("slot5") { Actions { broadcast("small5") deleteEntity() } }
        }
    }

    Entity {
        id("dudBall")
        template(true)
        startPosition(0, 0)
        color("475569")
        Type { circle(40) }
        Movement { speed(4.25) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("slot1") { Actions { broadcast("bonus1") burst(6, "475569") deleteEntity() } }
            OnCollision("slot2") { Actions { broadcast("bonus2") burst(6, "475569") deleteEntity() } }
            OnCollision("slot3") { Actions { broadcast("bonus3") burst(6, "475569") deleteEntity() } }
            OnCollision("slot4") { Actions { broadcast("bonus4") burst(6, "475569") deleteEntity() } }
            OnCollision("slot5") { Actions { broadcast("bonus5") burst(6, "475569") deleteEntity() } }
        }
    }

    Entity {
        id("spin")
        startPosition(300, 1080)
        color("fbbf24")
        Type { rectangle(200, 96) }
        Events {
            OnTapped("spin") {
                Actions {
                    whenBar("bet", "==", 1) { Actions {
                        sound("bell")
                        updateBar("bet", 1)
                        flash("ffd700", 0.3)
                        burst(12, "fbbf24")
                        shake(3, 0.2)
                        randomAction {
                            Actions { spawn("jackpotBall", 60, 780) }
                            Actions { spawn("jackpotBall", 210, 780) }
                            Actions { spawn("jackpotBall", 360, 780) }
                            Actions { spawn("jackpotBall", 510, 780) }
                            Actions { spawn("jackpotBall", 660, 780) }
                        }
                        randomAction {
                            Actions { spawn("smallBall", 60, 880) }
                            Actions { spawn("smallBall", 210, 880) }
                            Actions { spawn("smallBall", 360, 880) }
                            Actions { spawn("smallBall", 510, 880) }
                            Actions { spawn("smallBall", 660, 880) }
                        }
                        randomAction {
                            Actions { spawn("dudBall", 60, 980) }
                            Actions { spawn("dudBall", 210, 980) }
                            Actions { spawn("dudBall", 360, 980) }
                            Actions { spawn("dudBall", 510, 980) }
                            Actions { spawn("dudBall", 660, 980) }
                        }
                        delay(2.5) { Actions {
                            sound("lose")
                            flash("ff4444", 0.4)
                            burst(8, "ef4444")
                            gameOver("-- BUST --")
                        } }
                    } }
                }
            }
        }
    }
}
