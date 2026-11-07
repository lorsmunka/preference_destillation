Window {
    size(8, 8)
    background("020617")
    music("bg_6")
    gravity(0)

    lives(3, "-- VAPORISED --", "lose")
    Bar("storm") { initial(0) }
    winAt("storm", 4, "-- THE SUN QUIETS DOWN --", "win2")

    # four flare storms; a short Wave between them marks each one survived
    Wave(10, 1.2) { Actions { broadcast("flareH") } }
    Wave(1) { Actions { updateBar("storm", 1) sound("bell") } }
    Wave(12, 1.0) {
        Actions {
            randomAction {
                Actions { broadcast("flareH") }
                Actions { broadcast("flareV") }
            }
        }
    }
    Wave(1) { Actions { updateBar("storm", 1) sound("bell") } }
    Wave(14, 0.9) { Actions { broadcast("flareH") broadcast("flareV") } }
    Wave(1) { Actions { updateBar("storm", 1) sound("bell") } }
    Wave(16, 0.8) {
        Actions {
            randomAction {
                Actions { broadcast("flareH") broadcast("lockon") }
                Actions { broadcast("flareV") broadcast("flareH") }
                Actions { broadcast("flareV") }
            }
        }
    }
    Wave(2) { Actions { updateBar("storm", 1) } }

    Entity {
        id("sun")
        startPosition(680, 680)
        color("f59e0b")
        Type { circle(60) }
        Events {
            OnSignal("flareH") {
                Actions {
                    spawn("warnH", 0, random(0, 760))
                    flash("fde047", 0.2)
                }
            }
            OnSignal("flareV") {
                Actions {
                    spawn("warnV", random(0, 760), 0)
                    flash("fde047", 0.2)
                }
            }
            OnCollision("probe") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("probe", "ff0000", 0.3)
                    burst("probe", 16, "f59e0b")
                    shake(6, 0.3)
                }
            }
        }
    }

    Entity {
        id("probe")
        startPosition(80, 80)
        color("67e8f9")
        Type { diamond(34, 34) }
        Movement { speed(7) Control { movementControl(0) } }
        Events {
            OnSignal("lockon") {
                Actions {
                    sound("beep")
                    spawnHere("warnH")
                    spawnHere("warnV")
                }
            }
        }
    }

    Entity { id("warnH") template(true) startPosition(0, 0) color("22f97316")
        Type { rectangle(800, 40) }
        Events {
            OnElapsedTime(0.45) { Actions { setColor("66f97316") } }
            OnElapsedTime(0.9) { Actions { spawnHere("beamH") deleteEntity() } }
        } }

    Entity { id("warnV") template(true) startPosition(0, 0) color("22f97316")
        Type { rectangle(40, 800) }
        Events {
            OnElapsedTime(0.45) { Actions { setColor("66f97316") } }
            OnElapsedTime(0.9) { Actions { spawnHere("beamV") deleteEntity() } }
        } }

    Entity { id("beamH") template(true) startPosition(0, 0) color("fde047")
        Type { rectangle(800, 40) }
        Events {
            OnElapsedTime { Actions { sound("laser") shake(3, 0.15) } }
            OnCollision("probe") { Actions {
                sound("hurt") updateBar("lives", -1)
                flash("probe", "ff0000", 0.3) burst("probe", 14, "fde047") } }
            OnElapsedTime(0.4) { Actions { deleteEntity() } }
        } }

    Entity { id("beamV") template(true) startPosition(0, 0) color("fde047")
        Type { rectangle(40, 800) }
        Events {
            OnElapsedTime { Actions { sound("laser") shake(3, 0.15) } }
            OnCollision("probe") { Actions {
                sound("hurt") updateBar("lives", -1)
                flash("probe", "ff0000", 0.3) burst("probe", 14, "fde047") } }
            OnElapsedTime(0.4) { Actions { deleteEntity() } }
        } }
}
