Window {
    size(12, 8)
    background("0e7490")
    music("bg_4")
    gravity(0)

    winAt("sunk", 12, "-- THE BAY IS OURS --", "win")
    lives(3, "-- SENT TO THE DEEP --", "lose")
    Bar("gold") {
        initial(0)
        OnValue(3) { Actions { updateBar("lives", 1) sound("coin") flash("brig", "fde047", 0.4) } }
        OnValue(6) { Actions { updateBar("lives", 1) sound("coin") flash("brig", "fde047", 0.4) } }
    }

    Wave(24, 2.6) { Actions { spawn("galleon", 1070, random(20, 700)) } }
    Wave(30, 2.0) {
        Actions {
            randomAction {
                Actions { spawn("galleon", 1070, random(20, 700)) }
                Actions { spawn("galleon", 1070, random(20, 700)) }
                Actions { spawn("merchant", 1100, random(20, 720)) }
            }
        }
    }
    Wave(600, 1.5) {
        Actions {
            randomAction {
                Actions { spawn("galleon", 1070, random(20, 700)) }
                Actions { spawn("galleon", 1070, random(20, 360)) spawn("galleon", 1070, random(380, 700)) }
                Actions { spawn("merchant", 1100, random(20, 720)) }
            }
        }
    }

    Entity {
        id("brig")
        startPosition(30, 360)
        color("92400e")
        Type { rectangle(120, 70) }
        var("rapid", 0)
        Movement { speed(6) Control { movementControl(1) } }
        Events {
            OnElapsedTime(0.9) { repeatCount(-1) Actions { spawnHere("ball") sound("bump") } }
            OnElapsedTime(0.3) {
                repeatCount(-1)
                Actions { whenVar("rapid", ">", 0) { Actions { spawnHere("ball") } } }
            }
            OnSignal("rapid") {
                Actions {
                    changeVar("rapid", 1)
                    sound("pickup")
                    flash("fbbf24", 0.5)
                    delay(6) { Actions { changeVar("rapid", -1) } }
                }
            }
        }
    }

    Entity {
        id("ball")
        template(true)
        color("1f2937")
        Type { circle(10) }
        Movement { speed(7) AutoMove { direction(1, 0) } }
        Events { OnCollision { Actions { deleteEntity() } } }
    }

    Entity {
        id("galleon")
        template(true)
        color("7f1d1d")
        Type { rectangle(130, 70) }
        var("hp", 3)
        var("sinking", 0)
        Movement { speed(0.6) AutoMove { direction(-1, 0) } }
        Events {
            OnElapsedTime(random(2.5, 4)) {
                repeatCount(-1)
                Actions { whenVar("sinking", "==", 0) { Actions { spawnHere("grape") } } }
            }
            OnCollision("ball") {
                Actions {
                    whenVar("sinking", "==", 0) {
                        Actions {
                            changeVar("hp", -1)
                            deleteCollider()
                            flash("ffffff", 0.12)
                            burst(6, "fca5a5")
                            whenVar("hp", "==", 0) {
                                Actions {
                                    changeVar("sinking", 1)
                                    updateBar("sunk", 1)
                                    sound("glass")
                                    burst(20, "fbbf24")
                                    setColor("44403c")
                                    setOpacity(0.45)
                                    setVelocity(0, -45)
                                    delay(1.4) { Actions { deleteEntity() } }
                                }
                            }
                        }
                    }
                }
            }
            OnCollision("brig") {
                Actions {
                    whenVar("sinking", "==", 0) {
                        Actions {
                            sound("hurt")
                            updateBar("lives", -1)
                            flash("brig", "ff0000", 0.3)
                            shake(8, 0.4)
                            burst(18, "f97316")
                            deleteEntity()
                        }
                    }
                }
            }
            OnCollision {
                Actions {
                    whenVar("sinking", "==", 0) {
                        Actions {
                            sound("buzz")
                            updateBar("lives", -1)
                            shake(5, 0.3)
                            deleteEntity()
                        }
                    }
                }
            }
        }
    }

    Entity {
        id("merchant")
        template(true)
        color("f5f5f4")
        Type { rectangle(100, 50) }
        Movement { speed(0.9) AutoMove { direction(-1, 0) } }
        Events {
            OnCollision("ball") {
                Actions {
                    deleteCollider()
                    sound("buzz")
                    updateBar("lives", -1)
                    flash("brig", "ef4444", 0.4)
                    burst(14, "f5f5f4")
                    deleteEntity()
                }
            }
            OnCollision { Actions { updateBar("gold", 1) sound("coin") deleteEntity() } }
        }
    }

    Entity {
        id("grape")
        template(true)
        color("e5e7eb")
        Type { circle(8) }
        Movement { speed(3.5) AutoMove { direction(-1, 0) } }
        Events {
            OnCollision("brig") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("brig", "ff0000", 0.25)
                    shake(5, 0.25)
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("keg")
        template(true)
        color("b45309")
        Type { circle(18) }
        Movement { speed(1.2) AutoMove { direction(-1, 0) } }
        Events {
            OnCollision("brig") { Actions { broadcast("rapid") burst(10, "fbbf24") deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("lookout")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(13) { repeatCount(-1) Actions { spawn("keg", 1160, random(40, 720)) } }
        }
    }
}
