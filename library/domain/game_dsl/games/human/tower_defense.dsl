Window {
    size(8, 12)
    background("0c1525")
    gravity(0)

    lives(5, "-- BREACHED --", "lose")
    Bar("kills") { initial(0) }

    Entity { id("pathSeg1") startPosition(77, 1114) color("1f2937")
        Type { rectangle(640, 6) } }
    Entity { id("pathSeg2") startPosition(714, 717) color("1f2937")
        Type { rectangle(6, 400) } }
    Entity { id("pathSeg3") startPosition(77, 714) color("1f2937")
        Type { rectangle(640, 6) } }
    Entity { id("pathSeg4") startPosition(74, 317) color("1f2937")
        Type { rectangle(6, 400) } }

    Entity { id("base") startPosition(60, 250) color("fbbf24")
        Type { star(5, 45, 20) } }

    Entity {
        id("grunt")
        template(true)
        startPosition(0, 0)
        color("ef4444")
        Type { circle(34) }
        var("hp", 1)
        Movement { speed(2.55) Path {
            waypoint(60, 1100) waypoint(700, 1100)
            waypoint(700, 700) waypoint(60, 700)
            waypoint(60, 300) } }
        Events {
            OnTapped {
                Actions {
                    sound("click")
                    changeVar("hp", -1)
                    whenVar("hp", "<=", 0) { Actions {
                        sound("pop")
                        updateBar("kills", 1)
                        burst(12, "ef4444")
                        deleteEntity() } }
                }
            }
            OnCollision("base") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    shake(6, 0.3)
                    flash("base", "ff0000", 0.25)
                    burst(14, "fbbf24")
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("brute")
        template(true)
        startPosition(0, 0)
        color("991b1b")
        Type { circle(44) }
        var("hp", 3)
        Movement { speed(1.7) Path {
            waypoint(60, 1100) waypoint(700, 1100)
            waypoint(700, 700) waypoint(60, 700)
            waypoint(60, 300) } }
        Events {
            OnTapped {
                Actions {
                    sound("click")
                    changeVar("hp", -1)
                    whenVar("hp", "==", 2) { Actions {
                        setColor("dc2626")
                        flash("ffffff", 0.12) } }
                    whenVar("hp", "==", 1) { Actions {
                        setColor("f87171")
                        flash("ffffff", 0.12) } }
                    whenVar("hp", "<=", 0) { Actions {
                        sound("thud")
                        updateBar("kills", 1)
                        burst(16, "991b1b")
                        deleteEntity() } }
                }
            }
            OnCollision("base") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    shake(8, 0.4)
                    flash("base", "ff0000", 0.3)
                    burst(18, "fbbf24")
                    deleteEntity()
                }
            }
        }
    }

    Wave(6, 1.5) { Actions {
        spawn("grunt", 60, 1100) } }
    Wave(8, 1.2) { Actions {
        spawn("grunt", 60, 1100)
        randomAction {
            Actions { spawn("grunt", 60, 1100) }
            Actions { spawn("brute", 60, 1100) } } } }
    Wave(10, 0.9) { Actions {
        spawn("brute", 60, 1100) } }
    Wave(13) { Actions {
        sound("win")
        burst("base", 20, "fbbf24")
        shake(5, 0.5)
        gameOver("-- HELD THE BASE --") } }
}
