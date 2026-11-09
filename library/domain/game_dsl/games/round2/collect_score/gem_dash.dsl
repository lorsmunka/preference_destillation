Window {
    size(8, 12)
    background("1e1b4b")
    music("bg_5")
    gravity(0)

    Bar("level") {
        initial(1)
        OnValue(4) { Actions { broadcast("faster") } }
        OnValue(8) { Actions { broadcast("fastest") } }
    }
    Bar("score") { initial(0) }
    winAt("score", 40, "-- GEM MASTER --", "win2")
    lives(3, "-- THE MINE COLLAPSED --", "lose")

    Entity {
        id("player")
        startPosition(350, 50)
        color("a78bfa")
        Type { rectangle(100, 36) }
        Movement { speed(6) Control { movementControl(2) } }
        Events {
            OnElapsedTime { Actions { ramp("level", 1, 10, 60) } }
            OnSignal("faster") { Actions { flash("fde047", 0.4) sound("bell") } }
            OnSignal("fastest") { Actions { flash("f43f5e", 0.4) sound("bell") } }
        }
    }

    Entity {
        id("gem")
        template(true)
        startPosition(0, 0)
        color("4ade80")
        Type { diamond(36, 44) }
        Movement { speed(2.4) AutoMove { direction(0, -1) } }
        Events {
            OnSignal("fastest") { Actions { updateSpeed(4) } }
            OnCollision("player") {
                Actions {
                    sound("pickup")
                    updateBar("score", 1)
                    burst(8, "4ade80")
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("ruby")
        template(true)
        startPosition(0, 0)
        color("fb7185")
        Type { diamond(30, 54) }
        Movement { speed(3.6) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("player") {
                Actions {
                    sound("sparkle")
                    loopCount(3)
                    updateBar("score", 1)
                    burst(14, "fb7185")
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("bomb")
        template(true)
        startPosition(0, 0)
        color("64748b")
        Type { circle(24) }
        Movement { speed(2.8) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("player") {
                Actions {
                    sound("hurt")
                    shake(6, 0.3)
                    updateBar("lives", -1)
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("rubyMaker")
        template(true)
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(2.2) {
                repeatCount(-1)
                Actions { spawn("ruby", random(40, 720), 1160) }
            }
        }
    }

    Entity {
        id("director")
        startPosition(10, 10)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSignal("faster") { Actions { spawn("rubyMaker", 0, 0) } }
            OnSignal("fastest") {
                Actions {
                    delay(0.5) { Actions { spawn("bomb", random(40, 720), 1160) } }
                    delay(1.5) { Actions { spawn("bomb", random(40, 720), 1160) } }
                }
            }
        }
    }

    Wave(15, 1.2) { Actions { spawn("gem", random(40, 720), 1160) } }
    Wave(20, 0.9) {
        Actions {
            spawn("gem", random(40, 720), 1160)
            spawn("bomb", random(40, 720), 1160)
        }
    }
    Wave(40, 0.6) { Actions { spawn("gem", random(40, 720), 1160) } }
}
