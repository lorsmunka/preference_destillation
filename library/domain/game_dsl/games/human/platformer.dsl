Window {
    size(6, 12)
    background("0a0e27")
    music("bg_ambient")
    gravity(700)

    Entity {
        id("hero")
        startPosition(280, 120)
        color("4ade80")
        Type { rectangle(40, 50) }
        Movement { speed(6.8) Control { movementControl(2) } }
        Events {
            OnCollision("flag") {
                Actions {
                    sound("win")
                    flash("fbbf24", 0.4)
                    burst(20, "fbbf24")
                    shake(6, 0.4)
                    gameOver("-- REACHED THE TOP --")
                }
            }
        }
    }

    Entity { id("floor") startPosition(0, 70) color("64748b")
        Type { rectangle(600, 20) } oneWay(true) }

    Entity { id("p1") startPosition(60, 280) color("f97316")
        Type { rectangle(220, 16) } oneWay(true) }
    Entity { id("p2") startPosition(320, 420) color("f97316")
        Type { rectangle(220, 16) } oneWay(true) }
    Entity { id("p3") startPosition(60, 560) color("f97316")
        Type { rectangle(220, 16) } oneWay(true) }
    Entity { id("p4") startPosition(320, 700) color("f97316")
        Type { rectangle(220, 16) } oneWay(true) }
    Entity { id("p5") startPosition(60, 840) color("f97316")
        Type { rectangle(220, 16) } oneWay(true) }
    Entity { id("p6") startPosition(320, 980) color("f97316")
        Type { rectangle(220, 16) } oneWay(true) }

    Entity { id("flag") startPosition(260, 1110) color("fbbf24")
        Type { star(5, 40, 18) } }

    Entity {
        id("padL")
        startPosition(20, 20)
        color("3b82f6")
        Type { rectangle(160, 60) }
        Events {
            OnTapped("padL") {
                Actions {
                    sound("pop")
                    setVelocity("hero", -180, 520)
                    flash("ffffff", 0.1)
                    burst(6, "3b82f6")
                }
            }
        }
    }
    Entity { id("padLlbl") startPosition(60, 35) color("0a0e27")
        Type { text("LEFT", 26) } }

    Entity {
        id("padR")
        startPosition(420, 20)
        color("3b82f6")
        Type { rectangle(160, 60) }
        Events {
            OnTapped("padR") {
                Actions {
                    sound("pop")
                    setVelocity("hero", 180, 520)
                    flash("ffffff", 0.1)
                    burst(6, "3b82f6")
                }
            }
        }
    }
    Entity { id("padRlbl") startPosition(450, 35) color("0a0e27")
        Type { text("RIGHT", 26) } }
}
