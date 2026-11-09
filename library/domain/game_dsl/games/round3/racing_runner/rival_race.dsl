Window {
    size(8, 9)
    worldSize(40, 9)
    cameraFollow("car")
    background("1e293b")
    music("bg_6")
    gravity(0)

    lives(3, "-- CRASHED OUT --", "lose")

    Entity { id("kerbTop") startPosition(0, 820) color("475569")
        Type { rectangle(4000, 20) } }
    Entity { id("kerbBottom") startPosition(0, 40) color("475569")
        Type { rectangle(4000, 20) } }

    Entity { id("finish") startPosition(3800, 60) color("f8fafc")
        Type { rectangle(40, 760) } }

    Entity {
        id("car")
        startPosition(100, 500)
        color("22d3ee")
        Type { rectangle(90, 44) }
        Movement { speed(2.5) Control { movementControl(1) } }
        Events {
            OnCollision("cone") {
                Actions {
                    sound("hurt")
                    shake(6, 0.3)
                    flash("ff0000", 0.25)
                    updateBar("lives", -1)
                    deleteCollider()
                }
            }
            OnCollision("finish") {
                Actions {
                    sound("win")
                    shake(8, 0.6)
                    gameOver("-- YOU BEAT THE RIVAL --")
                }
            }
        }
    }

    Entity { id("pusher") startPosition(20, 20) color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.1) { repeatCount(-1) Actions {
                setVelocity("car", 400, 0)
            } }
        } }

    Entity {
        id("rival")
        startPosition(100, 200)
        color("f43f5e")
        Type { rectangle(90, 44) }
        Movement { speed(3.8) AutoMove { direction(1, 0) } }
        Events {
            OnCollision("finish") {
                Actions {
                    sound("lose")
                    gameOver("-- THE RIVAL TOOK IT --")
                }
            }
            OnElapsedTime(2) {
                repeatCount(-1)
                Actions { updateSpeed(random(3.2, 4.2)) }
            }
        }
    }

    Entity { id("cone") template(true) startPosition(0, 0) color("fb923c")
        Type { triangle(50, "up") } }

    Entity { id("layout") startPosition(40, 420) color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.05) { repeatCount(1) Actions {
                spawn("cone", 600, 500)
                spawn("cone", 900, 700)
                spawn("cone", 1200, 350)
                spawn("cone", 1500, 600)
                spawn("cone", 1500, 140)
                spawn("cone", 1900, 450)
                spawn("cone", 2200, 700)
                spawn("cone", 2500, 300)
                spawn("cone", 2800, 560)
                spawn("cone", 3100, 400)
                spawn("cone", 3400, 650)
            } }
        } }
}
