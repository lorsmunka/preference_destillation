Window {
    size(8, 12)
    background("0f172a")
    music("bg_5")
    gravity(0)

    Bar("fuel") {
        initial(12)
        OnValue(0) { Actions {
            sound("lose")
            gameOver("-- OUT OF FUEL --")
        } }
    }
    Bar("checkpoints") { initial(0) }
    winAt("checkpoints", 8, "-- ALL CHECKPOINTS --", "win")

    Entity { id("margin") startPosition(0, 0) color("1e293b")
        Type { rectangle(30, 1200) } }
    Entity { id("margin2") startPosition(770, 0) color("1e293b")
        Type { rectangle(30, 1200) } }

    Entity {
        id("car")
        startPosition(370, 100)
        color("4ade80")
        Type { rectangle(60, 100) }
        Movement { speed(6) Control { movementControl(0) } }
        Events {
            OnCollision("fuel") {
                Actions {
                    sound("pickup")
                    updateBar("fuel", 3)
                    deleteCollider()
                }
            }
            OnCollision("flag") {
                Actions {
                    sound("coin")
                    updateBar("checkpoints", 1)
                    burst(10, "facc15")
                    deleteCollider()
                }
            }
            OnCollision("oilDrum") {
                Actions {
                    sound("bump")
                    shake(6, 0.3)
                    updateBar("fuel", -2)
                    deleteCollider()
                }
            }
        }
    }

    Entity {
        id("fuel")
        template(true)
        startPosition(0, 0)
        color("f59e0b")
        Type { rectangle(36, 44) }
        Movement { speed(3) AutoMove { direction(0, -1) } }
        Events { OnCollision { Actions { deleteEntity() } } }
    }
    Entity {
        id("flag")
        template(true)
        startPosition(0, 0)
        color("e2e8f0")
        Type { star(5, 28, 12) }
        Movement { speed(3) AutoMove { direction(0, -1) } }
        Events { OnCollision { Actions { deleteEntity() } } }
    }
    Entity {
        id("oilDrum")
        template(true)
        startPosition(0, 0)
        color("7f1d1d")
        Type { circle(32) }
        Movement { speed(3.5) AutoMove { direction(0, -1) } }
        Events { OnCollision { Actions { deleteEntity() } } }
    }

    Entity {
        id("clock")
        startPosition(400, 20)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(1) {
                repeatCount(-1)
                Actions { updateBar("fuel", -1) }
            }
        }
    }

    Wave(40, 2.2) { Actions { spawn("fuel", random(60, 700), 1080) } }
    Wave(40, 3.5) { Actions { spawn("flag", random(60, 700), 1080) } }
    Wave(40, 1.6) { Actions { spawn("oilDrum", random(60, 700), 1080) } }
}
