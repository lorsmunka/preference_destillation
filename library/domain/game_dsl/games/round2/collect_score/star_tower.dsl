# Catch falling stars. Every score milestone unlocks something: bigger cart, golden stars, then a comet rain.
Window {
    size(9, 14)
    background("0f172a")
    music("bg_4")
    gravity(0)

    Bar("stars") {
        initial(0)
        OnValue(10) { Actions { sound("bell") broadcast("cartGrows") } }
        OnValue(20) { Actions { sound("bell") broadcast("goldRain") } }
        OnValue(35) { Actions { sound("bell") broadcast("cometRain") } }
    }
    winAt("stars", 50, "-- THE TOWER IS LIT --", "win2")
    Bar("lives") {
        initial(3)
        OnValue(0) { Actions { sound("lose") gameOver("-- THE SKY WENT DARK --") } }
    }

    Entity {
        id("cart")
        startPosition(390, 50)
        color("38bdf8")
        Type { rectangle(110, 30) }
        Movement { speed(6) Control { movementControl(2) } }
        Events {
            OnSignal("cartGrows") {
                Actions {
                    setScale(1.6)
                    flash("fde047", 0.5)
                }
            }
            OnSignal("cometRain") { Actions { updateSpeed(8) } }
        }
    }

    Entity {
        id("star")
        template(true)
        startPosition(0, 0)
        color("fde047")
        Type { star(5, 24, 11) }
        Movement { speed(2.6) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("cart") {
                Actions {
                    sound("pickup")
                    updateBar("stars", 1)
                    burst(8, "fde047")
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("goldStar")
        template(true)
        startPosition(0, 0)
        color("f59e0b")
        Type { star(6, 30, 12) }
        Movement { speed(3.8) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("cart") {
                Actions {
                    sound("sparkle")
                    loopCount(3)
                    updateBar("stars", 1)
                    burst(16, "f59e0b")
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("comet")
        template(true)
        startPosition(0, 0)
        color("f87171")
        Type { triangle(44, "down") }
        Movement { speed(4.2) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("cart") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    shake(7, 0.35)
                    burst(14, "f87171")
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("goldMaker")
        template(true)
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(2.4) {
                repeatCount(-1)
                Actions { spawn("goldStar", random(50, 850), 1360) }
            }
        }
    }

    Entity {
        id("cometMaker")
        template(true)
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(1.3) {
                repeatCount(-1)
                Actions { spawn("comet", random(50, 850), 1360) }
            }
        }
    }

    Entity {
        id("director")
        startPosition(10, 10)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSignal("goldRain") { Actions { spawn("goldMaker", 0, 0) } }
            OnSignal("cometRain") {
                Actions {
                    spawn("cometMaker", 0, 0)
                    shake(4, 0.5)
                }
            }
        }
    }

    Wave(12, 1.3) { Actions { spawn("star", random(50, 850), 1360) } }
    Wave(12, 0.9) { Actions { spawn("star", random(50, 850), 1360) } }
    Wave(60, 0.55) {
        Actions {
            randomAction {
                Actions { spawn("star", random(50, 850), 1360) }
                Actions { spawn("star", random(50, 850), 1360) }
                Actions { spawn("comet", random(50, 850), 1360) }
            }
        }
    }
}
