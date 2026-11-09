# Sidescroller-style dash across a wide arena: grab coins, boots make you fast for 5 seconds.
Window {
    size(12, 7)
    background("0f3d2e")
    music("bg_1")
    gravity(0)

    Bar("coins") { initial(0) }
    winAt("coins", 12, "-- ALL COINS DASHED UP --", "win")

    Entity {
        id("runner")
        startPosition(60, 300)
        color("fde68a")
        Type { circle(24) }
        Movement { speed(4) Control { movementControl(0) } }
        Events {
            OnSignal("bootsOn") {
                Actions {
                    updateSpeed(8)
                    flash("22d3ee", 0.3)
                    delay(5) { Actions { updateSpeed(4) } }
                }
            }
        }
    }

    Entity {
        id("boots")
        startPosition(560, 120)
        color("22d3ee")
        Type { triangle(40, "right") }
        Events {
            OnCollision("runner") {
                Actions {
                    sound("sparkle")
                    broadcast("bootsOn")
                    setEnable(false)
                    delay(12) { Actions { setEnable(true) } }
                }
            }
        }
    }

    Entity {
        id("coin")
        template(true)
        startPosition(0, 0)
        color("facc15")
        Type { circle(16) }
        Events {
            OnCollision("runner") {
                Actions {
                    sound("coin")
                    updateBar("coins", 1)
                    burst(6, "facc15")
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("spawner")
        startPosition(10, 10)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime { Actions { spawn("coin", 200, 560) spawn("coin", 900, 100) spawn("coin", 1100, 560) } }
            OnElapsedTime(2) {
                repeatCount(9)
                Actions { spawn("coin", random(120, 1080), random(80, 600)) }
            }
        }
    }

    Entity {
        id("sentry")
        startPosition(300, 100)
        color("ef4444")
        Type { rectangle(36, 36) }
        Movement { speed(3) Path { waypoint(300, 100) waypoint(300, 600) waypoint(800, 600) waypoint(800, 100) } }
        Events {
            OnCollision("runner") {
                Actions {
                    sound("hurt")
                    setPosition("runner", 60, 300, 0)
                    shake(5, 0.3)
                }
            }
        }
    }

    Entity {
        id("clock")
        startPosition(10, 10)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(45) {
                Actions {
                    whenBar("coins", "<", 12) { Actions { sound("lose") gameOver("-- TOO SLOW --") } }
                }
            }
        }
    }
}
