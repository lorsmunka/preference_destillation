Window {
    size(8, 9)
    worldSize(40, 9)
    cameraFollow("runner")
    background("0f172a")
    gravity(0)

    lives(3, "-- CAUGHT --", "lose")
    Bar("met") {
        initial(0)
        OnValue(1) { Actions {
            sound("win")
            shake(10, 0.8)
            burst("runner", 40, "fbbf24")
            gameOver("-- REACHED THE FLAG --")
        } }
    }

    Entity { id("floor") startPosition(0, 60) color("1e293b")
        Type { rectangle(4000, 40) } }

    Entity {
        id("runner")
        startPosition(200, 400)
        color("4ade80")
        Type { rectangle(60, 60) }
        Movement { speed(2.55) Control { movementControl(1) } }
        Events {
            OnCollision("barrier") {
                Actions {
                    sound("hurt")
                    shake(7, 0.4)
                    flash("ff0000", 0.3)
                    burst(16, "ef4444")
                    updateBar("lives", -1)
                    setPosition("runner", 200, 400, 0)
                }
            }
            OnCollision("flag") {
                Actions { updateBar("met", 1) }
            }
        }
    }

    Entity { id("pusher") startPosition(20, 20) color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.1) { repeatCount(-1) Actions {
                setVelocity("runner", 360, 0)
            } }
        } }

    Entity { id("barrier") template(true) startPosition(0, 0) color("ef4444")
        Type { rectangle(40, 300) } }

    Entity { id("flag") template(true) startPosition(0, 0) color("fbbf24")
        Type { rectangle(80, 500) } }
    Entity { id("layout") startPosition(40, 40) color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.05) { repeatCount(1) Actions {
                spawn("barrier", 600, 100)
                spawn("barrier", 900, 500)
                spawn("barrier", 1250, 100)
                spawn("barrier", 1550, 100)
                spawn("barrier", 1550, 600)
                spawn("barrier", 1900, 400)
                spawn("barrier", 2300, 100)
                spawn("barrier", 2650, 500)
                spawn("barrier", 3000, 100)
                spawn("barrier", 3000, 600)
                spawn("barrier", 3350, 250)
                spawn("flag", 3850, 200)
            } }
        } }
}
