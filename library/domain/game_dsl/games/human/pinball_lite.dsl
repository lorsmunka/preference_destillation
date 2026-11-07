Window {
    size(7, 12)
    background("0a0e27")
    gravity(450)

    lives(3, "-- TILT --", "lose")
    Bar("score") { initial(0) }

    Entity {
        id("ball")
        template(true)
        startPosition(0, 0)
        color("fbbf24")
        Type { circle(16) }
        Movement { speed(0.85) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("bumper1") { Actions { bounceBack() } }
            OnCollision("bumper2") { Actions { bounceBack() } }
            OnCollision("bumper3") { Actions { bounceBack() } }
            OnCollision("bumper4") { Actions { bounceBack() } }
            OnCollision("bumper5") { Actions { bounceBack() } }
            OnCollision("wallL") { Actions { bounceBack() } }
            OnCollision("wallR") { Actions { bounceBack() } }
            OnCollision("ceiling") { Actions { bounceBack() } }
            OnCollision("floorL") { Actions { bounceBack() } }
            OnCollision("floorR") { Actions { bounceBack() } }
            OnCollision("drain") {
                Actions {
                    sound("hurt")
                    shake(5, 0.35)
                    burst(16, "fbbf24")
                    updateBar("lives", -1)
                    deleteEntity()
                    spawn("ball", 350, 1100)
                }
            }
            OnSignal("won") {
                Actions {
                    sound("win")
                    shake(8, 0.6)
                    burst(40, "fbbf24")
                    gameOver("-- HIGH ROLLER --")
                }
            }
        }
    }

    Entity {
        id("director")
        startPosition(10, 10)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.05) {
                repeatCount(1)
                Actions {
                    spawn("ball", 350, 1100)
                }
            }
        }
    }

    Entity { id("wallL") startPosition(0, 100) color("1e293b")
        Type { rectangle(10, 1100) } solid(true) }
    Entity { id("wallR") startPosition(690, 100) color("1e293b")
        Type { rectangle(10, 1100) } solid(true) }
    Entity { id("ceiling") startPosition(0, 1190) color("1e293b")
        Type { rectangle(700, 10) } solid(true) }

    Entity { id("bumper1") startPosition(150, 880) color("ef4444")
        Type { circle(36) }
        Events { OnCollision("ball") {
            Actions {
                sound("pop")
                flash("ffffff", 0.15)
                burst(10, "ef4444")
                updateBar("score", 1)
                whenBar("score", ">=", 25) { Actions { broadcast("won") } }
            } } } }
    Entity { id("bumper2") startPosition(520, 880) color("3b82f6")
        Type { circle(36) }
        Events { OnCollision("ball") {
            Actions {
                sound("pop")
                flash("ffffff", 0.15)
                burst(10, "3b82f6")
                updateBar("score", 1)
                whenBar("score", ">=", 25) { Actions { broadcast("won") } }
            } } } }
    Entity { id("bumper3") startPosition(335, 740) color("a855f7")
        Type { star(5, 42, 20) }
        Events { OnCollision("ball") {
            Actions {
                sound("bell")
                flash("ffffff", 0.15)
                burst(12, "a855f7")
                updateBar("score", 2)
                whenBar("score", ">=", 25) { Actions { broadcast("won") } }
            } } } }
    Entity { id("bumper4") startPosition(200, 580) color("10b981")
        Type { circle(32) }
        Events { OnCollision("ball") {
            Actions {
                sound("pop")
                flash("ffffff", 0.15)
                burst(10, "10b981")
                updateBar("score", 1)
                whenBar("score", ">=", 25) { Actions { broadcast("won") } }
            } } } }
    Entity { id("bumper5") startPosition(470, 580) color("f59e0b")
        Type { circle(32) }
        Events { OnCollision("ball") {
            Actions {
                sound("pop")
                flash("ffffff", 0.15)
                burst(10, "f59e0b")
                updateBar("score", 1)
                whenBar("score", ">=", 25) { Actions { broadcast("won") } }
            } } } }

    Entity { id("drain") startPosition(220, 0) color("450a0a")
        Type { rectangle(260, 12) } }

    Entity { id("floorL") startPosition(0, 0) color("1e293b")
        Type { rectangle(220, 12) } }
    Entity { id("floorR") startPosition(480, 0) color("1e293b")
        Type { rectangle(220, 12) } }

    Entity {
        id("flipL")
        startPosition(20, 40)
        color("4ade80")
        Type { rectangle(180, 70) }
        Events {
            OnTapped("flipL") {
                Actions {
                    sound("click")
                    applyImpulse("ball", 220, 480)
                    flash("ffffff", 0.1)
                    burst(6, "4ade80")
                }
            }
        }
    }

    Entity {
        id("flipR")
        startPosition(500, 40)
        color("4ade80")
        Type { rectangle(180, 70) }
        Events {
            OnTapped("flipR") {
                Actions {
                    sound("click")
                    applyImpulse("ball", -220, 480)
                    flash("ffffff", 0.1)
                    burst(6, "4ade80")
                }
            }
        }
    }

    Entity { id("flipLlbl") startPosition(80, 64) color("0a0e27")
        Type { text("LEFT", 28) } }
    Entity { id("flipRlbl") startPosition(555, 64) color("0a0e27")
        Type { text("RIGHT", 28) } }
}
