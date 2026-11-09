Window {
    size(6, 12)
    background("1e3a5f")
    music("bg_2")
    gravity(700)
    lives(3, "OUT OF LIVES")

    Entity {
        id("hero")
        startPosition(60, 100)
        color("4ade80")
        Type { rectangle(36, 46) }
        Movement { speed(6) Control { movementControl(2) } }
        Events {
            OnCollision("spike") {
                Actions {
                    sound("hurt")
                    flash("ef4444", 0.3)
                    updateBar("lives", -1)
                    setPosition("hero", 60, 100, 0)
                }
            }
        }
    }

    Entity { id("ground") startPosition(0, 0) color("334155")
        Type { rectangle(600, 80) } solid(true) }

    Entity { id("platform1") startPosition(40, 260) color("f97316")
        Type { rectangle(200, 16) } oneWay(true) }
    Entity { id("platform2") startPosition(340, 400) color("f97316")
        Type { rectangle(200, 16) } oneWay(true) }
    Entity { id("platform3") startPosition(60, 540) color("f97316")
        Type { rectangle(200, 16) } oneWay(true) }
    Entity { id("platform4") startPosition(340, 680) color("f97316")
        Type { rectangle(200, 16) } oneWay(true) }
    Entity { id("platform5") startPosition(60, 820) color("f97316")
        Type { rectangle(200, 16) } oneWay(true) }

    Entity { id("spike") startPosition(420, 416) color("ef4444")
        Type { triangle(30, "up") } }
    Entity { id("spike2") startPosition(150, 556) color("ef4444")
        Type { triangle(30, "up") }
        Events { OnCollision("hero") { Actions {
            sound("hurt")
            updateBar("lives", -1)
            setPosition("hero", 60, 100, 0)
        } } } }

    goal("flag", 100, 840, 80, 50, "YOU MADE IT")

    Entity {
        id("jumpButton")
        startPosition(200, 10)
        color("3b82f6")
        Type { rectangle(200, 60) }
        Events {
            OnTapped("jumpButton") {
                Actions {
                    sound("jump")
                    applyImpulse("hero", 0, 560)
                }
            }
        }
    }
    Entity { id("jumpLabel") startPosition(245, 25) color("ffffff")
        Type { text("JUMP", 30) } }
}
