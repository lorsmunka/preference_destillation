Window {
    size(8, 9)
    worldSize(16, 18)
    cameraFollow("hero")
    background("0a0e27")
    music("bg_ambient")
    gravity(0)

    Bar("coins") {
        initial(0)
        OnValue(5) {
            Actions {
                sound("win")
                shake(10, 0.8)
                burst("hero", 40, "fbbf24")
                gameOver("-- WORLD MAPPED --")
            }
        }
    }

    Entity { id("q1") startPosition(0, 0) color("0f172a")
        Type { rectangle(800, 900) } }
    Entity { id("q2") startPosition(800, 0) color("0c1525")
        Type { rectangle(800, 900) } }
    Entity { id("q3") startPosition(0, 900) color("0c1525")
        Type { rectangle(800, 900) } }
    Entity { id("q4") startPosition(800, 900) color("111b30")
        Type { rectangle(800, 900) } }

    Entity { id("wall1") startPosition(400, 200) color("64748b")
        Type { rectangle(40, 400) } solid(true) }
    Entity { id("wall2") startPosition(800, 1000) color("64748b")
        Type { rectangle(400, 40) } solid(true) }
    Entity { id("wall3") startPosition(1200, 400) color("64748b")
        Type { rectangle(40, 500) } solid(true) }
    Entity { id("wall4") startPosition(200, 1200) color("64748b")
        Type { rectangle(400, 40) } solid(true) }

    Entity {
        id("hero")
        startPosition(120, 120)
        color("22d3ee")
        Type { rectangle(60, 60) }
        Movement { speed(6.8) Control { movementControl(0) } }
        Events {
            OnCollision("coin") {
                Actions {
                    sound("coin")
                    flash("fbbf24", 0.2)
                    burst(14, "fbbf24")
                    updateBar("coins", 1)
                    deleteCollider()
                }
            }
        }
    }

    Entity { id("coin") template(true) startPosition(0, 0) color("fbbf24")
        Type { star(5, 20, 9) } }
    Entity { id("layout") startPosition(20, 20) color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.05) {
                repeatCount(1)
                Actions {
                    spawn("coin", 300, 1500)
                    spawn("coin", 1400, 200)
                    spawn("coin", 1100, 1300)
                    spawn("coin", 600, 700)
                    spawn("coin", 1500, 1600)
                }
            }
        } }
}
