Window {
    size(8, 12)
    background("0b1020")
    music("bg_3")
    gravity(0)

    Bar("boss") {
        initial(12)
        OnValue(6) { Actions { broadcast("enraged") } }
        OnValue(0) {
            Actions {
                sound("win")
                shake(8, 0.6)
                burst("boss", 20, "a855f7")
                gameOver("-- BOSS DOWN --")
            }
        }
    }
    lives(3, "-- DEFEATED --", "lose")

    Entity {
        id("player")
        startPosition(360, 90)
        color("22d3ee")
        Type { circle(40) }
        Movement { speed(6.8) Control { movementControl(0) } }
    }

    Entity {
        id("boss")
        startPosition(280, 1010)
        color("a855f7")
        Type { star(6, 120, 55) }
        Events {
            OnCollision("player") {
                Actions {
                    sound("thud")
                    updateBar("boss", -1)
                    flash("ffffff", 0.15)
                    burst(10, "a855f7")
                }
            }
            Sequence {
                Phase(3, 2.8) {
                    Actions {
                        setColor("ef4444")
                        delay(0.7) {
                            Actions {
                                sound("laser")
                                spawn("homing", 400, 940)
                                spawn("shot", 250, 980)
                                spawn("shot", 540, 980)
                                setColor("a855f7")
                            }
                        }
                    }
                }
                Phase(-1, 1.7) {
                    Actions {
                        setColor("ef4444")
                        delay(0.5) {
                            Actions {
                                sound("laser")
                                spawn("homing", 320, 940)
                                spawn("homing", 480, 940)
                                spawn("shot", 250, 980)
                                spawn("shot", 540, 980)
                                setColor("a855f7")
                            }
                        }
                    }
                }
            }
            OnSignal("enraged") {
                Actions {
                    sound("buzz")
                    shake(7, 0.4)
                    flash("f97316", 0.5)
                    spawn("homing", 300, 940)
                    spawn("homing", 500, 940)
                }
            }
            OnElapsedTime(2.0) {
                repeatCount(-1)
                Actions {
                    whenBar("boss", "<=", 6) {
                        Actions {
                            flash("ff0000", 0.25)
                        }
                    }
                }
            }
        }
    }

    Entity {
        id("homing")
        template(true)
        startPosition(0, 0)
        color("f97316")
        Type { circle(24) }
        Movement { speed(1.7) chase("player") }
        Events {
            OnCollision("player") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("player", "ff0000", 0.25)
                    shake(6, 0.3)
                    burst(12, "ff6b6b")
                    deleteEntity()
                }
            }
            OnCollision("boss") {
                Actions {
                    sound("pop")
                    burst(8, "a855f7")
                    deleteEntity()
                }
            }
            OnElapsedTime(8.0) {
                Actions {
                    setColor("ef4444") retarget("boss") updateBar("boss", -1)
                }
            }
        }
    }

    Entity {
        id("shot")
        template(true)
        startPosition(0, 0)
        color("fbbf24")
        Type { circle(16) }
        Movement { speed(4.25) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("player") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("player", "ff0000", 0.25)
                    shake(5, 0.25)
                    burst(10, "fbbf24")
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }
}
