Window {
    size(7, 12)
    background("0a0e27")
    music("bg_1")
    gravity(0)

    Bar("boxes") {
        initial(0)
        OnValue(3) {
            Actions {
                sound("win")
                shake(8, 0.6)
                burst("hero", 40, "fbbf24")
                gameOver("-- WAREHOUSE CLEARED --")
            }
        }
    }

    Entity { id("wallT") startPosition(50, 900) color("1e293b")
        Type { rectangle(600, 20) } solid(true) }
    Entity { id("wallB") startPosition(50, 280) color("1e293b")
        Type { rectangle(600, 20) } solid(true) }
    Entity { id("wallL") startPosition(30, 280) color("1e293b")
        Type { rectangle(20, 640) } solid(true) }
    Entity { id("wallR") startPosition(650, 280) color("1e293b")
        Type { rectangle(20, 640) } solid(true) }

    Entity { id("innerW1") startPosition(250, 700) color("1e293b")
        Type { rectangle(100, 100) } solid(true) }
    Entity { id("innerW2") startPosition(450, 500) color("1e293b")
        Type { rectangle(100, 100) } solid(true) }

    Entity { id("target1") startPosition(550, 800) color("78350f")
        Type { diamond(100, 100) } }
    Entity { id("target2") startPosition(150, 600) color("78350f")
        Type { diamond(100, 100) } }
    Entity { id("target3") startPosition(150, 400) color("78350f")
        Type { diamond(100, 100) } }

    Entity { id("box1") startPosition(350, 800) color("a16207")
        Type { rectangle(100, 100) } pushable(true) var("docked", 0)
        Events {
            OnCollision("target1") {
                Actions {
                    whenVar("docked", "==", 0) {
                        Actions {
                            sound("pickup")
                            changeVar("docked", 1)
                            updateBar("boxes", 1)
                            flash("fbbf24", 0.3)
                            burst(12, "fbbf24")
                        }
                    }
                }
            }
            OnCollision("target2") {
                Actions {
                    whenVar("docked", "==", 0) {
                        Actions {
                            sound("pickup")
                            changeVar("docked", 1)
                            updateBar("boxes", 1)
                            flash("fbbf24", 0.3)
                            burst(12, "fbbf24")
                        }
                    }
                }
            }
            OnCollision("target3") {
                Actions {
                    whenVar("docked", "==", 0) {
                        Actions {
                            sound("pickup")
                            changeVar("docked", 1)
                            updateBar("boxes", 1)
                            flash("fbbf24", 0.3)
                            burst(12, "fbbf24")
                        }
                    }
                }
            }
        }
    }
    Entity { id("box2") startPosition(250, 500) color("a16207")
        Type { rectangle(100, 100) } pushable(true) var("docked", 0)
        Events {
            OnCollision("target1") {
                Actions {
                    whenVar("docked", "==", 0) {
                        Actions {
                            sound("pickup")
                            changeVar("docked", 1)
                            updateBar("boxes", 1)
                            flash("fbbf24", 0.3)
                            burst(12, "fbbf24")
                        }
                    }
                }
            }
            OnCollision("target2") {
                Actions {
                    whenVar("docked", "==", 0) {
                        Actions {
                            sound("pickup")
                            changeVar("docked", 1)
                            updateBar("boxes", 1)
                            flash("fbbf24", 0.3)
                            burst(12, "fbbf24")
                        }
                    }
                }
            }
            OnCollision("target3") {
                Actions {
                    whenVar("docked", "==", 0) {
                        Actions {
                            sound("pickup")
                            changeVar("docked", 1)
                            updateBar("boxes", 1)
                            flash("fbbf24", 0.3)
                            burst(12, "fbbf24")
                        }
                    }
                }
            }
        }
    }
    Entity { id("box3") startPosition(450, 400) color("a16207")
        Type { rectangle(100, 100) } pushable(true) var("docked", 0)
        Events {
            OnCollision("target1") {
                Actions {
                    whenVar("docked", "==", 0) {
                        Actions {
                            sound("pickup")
                            changeVar("docked", 1)
                            updateBar("boxes", 1)
                            flash("fbbf24", 0.3)
                            burst(12, "fbbf24")
                        }
                    }
                }
            }
            OnCollision("target2") {
                Actions {
                    whenVar("docked", "==", 0) {
                        Actions {
                            sound("pickup")
                            changeVar("docked", 1)
                            updateBar("boxes", 1)
                            flash("fbbf24", 0.3)
                            burst(12, "fbbf24")
                        }
                    }
                }
            }
            OnCollision("target3") {
                Actions {
                    whenVar("docked", "==", 0) {
                        Actions {
                            sound("pickup")
                            changeVar("docked", 1)
                            updateBar("boxes", 1)
                            flash("fbbf24", 0.3)
                            burst(12, "fbbf24")
                        }
                    }
                }
            }
        }
    }

    Entity {
        id("hero")
        startPosition(350, 400)
        color("22d3ee")
        Type { rectangle(100, 100) }
        Movement { GridMove(100, 1){ direction(0, 0) } }
        Events {
            OnSignal("stepUp") { Actions { gridStep(0, 1) } }
            OnSignal("stepDown") { Actions { gridStep(0, -1) } }
            OnSignal("stepLeft") { Actions { gridStep(-1, 0) } }
            OnSignal("stepRight") { Actions { gridStep(1, 0) } }
        }
    }

    Entity { id("padUp") startPosition(290, 150) color("3b82f6")
        Type { rectangle(120, 100) }
        Events { OnTapped("padUp") { Actions { sound("bump") broadcast("stepUp")
            flash("ffffff", 0.08) } } } }
    Entity { id("padDown") startPosition(290, 30) color("3b82f6")
        Type { rectangle(120, 100) }
        Events { OnTapped("padDown") { Actions { sound("bump") broadcast("stepDown")
            flash("ffffff", 0.08) } } } }
    Entity { id("padLeft") startPosition(150, 90) color("3b82f6")
        Type { rectangle(120, 100) }
        Events { OnTapped("padLeft") { Actions { sound("bump") broadcast("stepLeft")
            flash("ffffff", 0.08) } } } }
    Entity { id("padRight") startPosition(430, 90) color("3b82f6")
        Type { rectangle(120, 100) }
        Events { OnTapped("padRight") { Actions { sound("bump") broadcast("stepRight")
            flash("ffffff", 0.08) } } } }
}
