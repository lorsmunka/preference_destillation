# Pot Drop: a mouse on a busy kitchen floor. Rings mark where the clumsy cook's pots will land.
Window {
    size(8, 10)
    background("3f2a1d")
    music("bg_1")
    gravity(0)

    lives(3, "-- SQUASHED --", "lose")
    Bar("time") {
        initial(50)
        OnValue(0) { Actions {
            sound("win")
            burst("mouse", 30, "fbbf24")
            shake(5, 0.4)
            gameOver("-- DINNER IS SERVED --")
        } }
    }

    joystick("mouse", 80, 5)

    Entity { id("mouse") startPosition(380, 480) color("cbd5e1")
        Type { circle(18) }
        Movement { speed(0) Control { movementControl(0) } }
    }

    Entity { id("counter") startPosition(0, 900) color("78350f") solid(true)
        Type { rectangle(800, 100) }
        Events {
            OnElapsedTime { Actions { ramp("time", 50, 0, 50) } }
            OnElapsedTime(1.1) {
                repeatCount(-1)
                Actions {
                    spawn("target", random(220, 700), random(40, 800))
                    whenBar("time", "<=", 35) { Actions { spawn("target", random(40, 700), random(220, 800)) } }
                    whenBar("time", "<=", 15) { Actions { spawn("target", random(40, 700), random(40, 800)) } }
                }
            }
            OnElapsedTime(2.4) {
                repeatCount(-1)
                Actions { whenBar("time", "<=", 25) { Actions { sound("laser") spawn("knife", 0, random(240, 820)) } } }
            }
            OnElapsedTime(12) { repeatCount(-1) Actions { spawn("cheese", random(240, 720), random(60, 820)) } }
        }
    }

    Entity { id("target") template(true) startPosition(0, 0) color("66fde68a")
        Type { ring(45, 34) }
        Events {
            OnElapsedTime(0.45) { Actions { setColor("bbf97316") } }
            OnElapsedTime(0.9) { Actions { spawnHere("pot") deleteEntity() } }
        }
    }

    Entity { id("pot") template(true) startPosition(0, 0) color("94a3b8")
        Type { circle(45) }
        Events {
            OnElapsedTime { Actions { sound("clang") shake(4, 0.15) burst(10, "e2e8f0") } }
            OnCollision("mouse") { Actions {
                sound("hurt")
                updateBar("lives", -1)
                flash("mouse", "ff0000", 0.4)
                burst("mouse", 14, "f87171")
                deleteEntity()
            } }
            OnElapsedTime(0.5) { Actions { deleteEntity() } }
        }
    }

    Entity { id("knife") template(true) startPosition(0, 0) color("e5e7eb")
        Type { rectangle(90, 14) }
        Movement { speed(5) AutoMove { direction(1, 0) } }
        Events {
            OnCollision("mouse") { Actions {
                sound("hurt")
                updateBar("lives", -1)
                flash("mouse", "ff0000", 0.4)
                shake(6, 0.25)
                deleteEntity()
            } }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity { id("cheese") template(true) startPosition(0, 0) color("facc15")
        Type { triangle(34, "up") }
        Events {
            OnCollision("mouse") { Actions {
                sound("pickup")
                burst(12, "facc15")
                whenBar("lives", "<", 3) { Actions { updateBar("lives", 1) flash("mouse", "4ade80", 0.3) } }
                deleteEntity()
            } }
            OnElapsedTime(6) { Actions { deleteEntity() } }
        }
    }
}
