# Fireflies blink a few times and then fly off. Tap them while they glow, but leave the moths alone.
Window {
    size(7, 11)
    background("0c1a2b")
    music("bg_nature")
    lives(3, "THE JAR STAYS EMPTY", "lose")
    winAt("score", 12, "JAR FULL OF LIGHT", "win")

    Entity { id("ground") startPosition(0, 0) color("14301f") Type { rectangle(700, 120) } }
    Entity { id("moon") startPosition(520, 930) color("fef9c3") Type { circle(50) } }

    Entity {
        id("firefly")
        template(true)
        startPosition(0, 0)
        color("fde047")
        Type { circle(32) }
        Events {
            Sequence {
                Phase(1, 0.3) { Actions { setOpacity(0.3) } }
                Phase(3, 0.5) { Actions {
                    setOpacity(1)
                    flash("fef9c3", 0.25)
                    delay(0.25) { Actions { setOpacity(0.3) } }
                } }
                Phase(1, 0.4) { Actions {
                    updateBar("lives", -1)
                    sound("hurt")
                    deleteEntity()
                } }
            }
            OnTapped { Actions {
                updateBar("score", 1)
                sound("sparkle")
                burst(10, "fde047")
                deleteEntity()
            } }
        }
    }

    Entity {
        id("moth")
        template(true)
        startPosition(0, 0)
        color("a8a29e")
        Type { triangle(50, "up") }
        Events {
            OnTapped { Actions {
                updateBar("lives", -1)
                sound("buzz")
                flash("ef4444", 0.2)
                deleteEntity()
            } }
            OnElapsedTime(2.5) { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("spawner")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(1.6) {
                repeatCount(-1)
                Actions { spawn("firefly", random(60, 580), random(220, 900)) }
            }
            OnElapsedTime(3.1) {
                repeatCount(-1)
                Actions { spawn("moth", random(60, 580), random(220, 900)) }
            }
        }
    }
}
