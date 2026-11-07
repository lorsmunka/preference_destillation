# Pin Dodge: you are the last pin standing. Tap a lane pad to hop into that lane before the ball arrives.
Window {
    size(6, 12)
    background("a16207")
    music("bg_4")
    gravity(0)

    lives(3, "-- KNOCKED DOWN --", "lose")
    Bar("frame") { initial(0) }
    winAt("frame", 10, "-- PERFECT GAME --", "win")

    Wave(8, 1.3) { Actions { broadcast("roll") } }
    Wave(8, 1.1) { Actions { broadcast("roll") } }
    Wave(10, 1.0) {
        Actions {
            randomAction {
                Actions { broadcast("roll") }
                Actions { broadcast("hook") }
            }
        }
    }
    Wave(600, 0.9) {
        Actions {
            randomAction {
                Actions { broadcast("roll") }
                Actions { broadcast("split") }
                Actions { broadcast("hook") }
            }
        }
    }

    Entity { id("gutter1") startPosition(195, 120) color("78350f") Type { rectangle(10, 1080) } }
    Entity { id("gutter2") startPosition(395, 120) color("78350f") Type { rectangle(10, 1080) } }

    Entity {
        id("pin")
        startPosition(280, 160)
        color("f8fafc")
        Type { rectangle(40, 70) }
    }

    Entity {
        id("padL")
        startPosition(0, 0)
        color("1e293b")
        Type { rectangle(195, 120) }
        Events {
            OnTapped("padL") {
                Actions {
                    sound("click")
                    setPosition("pin", 80, 160, 0.12)
                    flash("475569", 0.1)
                }
            }
        }
    }

    Entity {
        id("padM")
        startPosition(205, 0)
        color("1e293b")
        Type { rectangle(190, 120) }
        Events {
            OnTapped("padM") {
                Actions {
                    sound("click")
                    setPosition("pin", 280, 160, 0.12)
                    flash("475569", 0.1)
                }
            }
        }
    }

    Entity {
        id("padR")
        startPosition(405, 0)
        color("1e293b")
        Type { rectangle(195, 120) }
        Events {
            OnTapped("padR") {
                Actions {
                    sound("click")
                    setPosition("pin", 480, 160, 0.12)
                    flash("475569", 0.1)
                }
            }
        }
    }

    Entity {
        id("bowler")
        startPosition(0, 1170)
        color("312e81")
        Type { rectangle(600, 30) }
        Events {
            OnElapsedTime(5) { repeatCount(10) Actions { updateBar("frame", 1) sound("bell") } }
            OnSignal("roll") { Actions { randomAction {
                Actions { spawn("ball", 70, 1100) }
                Actions { spawn("ball", 270, 1100) }
                Actions { spawn("ball", 470, 1100) }
            } } }
            OnSignal("split") { Actions { sound("clang") randomAction {
                Actions { spawn("ball", 70, 1100) spawn("ball", 270, 1100) }
                Actions { spawn("ball", 270, 1100) spawn("ball", 470, 1100) }
                Actions { spawn("ball", 70, 1100) spawn("ball", 470, 1100) }
            } } }
            OnSignal("hook") { Actions { randomAction {
                Actions { spawn("hookR", 70, 1100) }
                Actions { spawn("hookL", 470, 1100) }
            } } }
        }
    }

    Entity { id("ball") template(true) startPosition(0, 0) color("111827") Type { circle(30) }
        Movement { speed(5) AutoMove { direction(0, -1) } }
        Events {
            OnElapsedTime { Actions {
                whenBar("frame", ">=", 5) { Actions { updateSpeed(6.5) } }
                whenBar("frame", ">=", 8) { Actions { updateSpeed(8) } } } }
            OnCollision("pin") { Actions { sound("thud") updateBar("lives", -1) flash("pin", "ef4444", 0.4)
                shake(8, 0.4) burst("pin", 16, "f8fafc") deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        } }

    Entity { id("hookR") template(true) startPosition(0, 0) color("7c3aed") Type { circle(30) }
        Movement { speed(4.5) AutoMove { direction(0.4, -1) } }
        Events {
            OnCollision("pin") { Actions { sound("thud") updateBar("lives", -1) flash("pin", "ef4444", 0.4)
                shake(8, 0.4) burst("pin", 16, "c4b5fd") deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        } }

    Entity { id("hookL") template(true) startPosition(0, 0) color("7c3aed") Type { circle(30) }
        Movement { speed(4.5) AutoMove { direction(-0.4, -1) } }
        Events {
            OnCollision("pin") { Actions { sound("thud") updateBar("lives", -1) flash("pin", "ef4444", 0.4)
                shake(8, 0.4) burst("pin", 16, "c4b5fd") deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        } }
}
