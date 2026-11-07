Window {
    size(9, 12)
    background("0f172a")
    music("bg_ambient")
    gravity(0)

    lives(3, "-- SWEPT AWAY --", "lose")
    Bar("pressure") { initial(0) }

    Entity {
        id("gull")
        startPosition(430, 580)
        color("f8fafc")
        Type { triangle(40, "right") }
        Movement { speed(5) Control { movementControl(0) } }
    }

    # the storm walls close in by growing (scale 2, then 3) and fall back when it calms
    Entity { id("stormL") startPosition(0, 0) color("9964748b") Type { rectangle(100, 1200) }
        Events {
            OnSignal("warn") { Actions { flash("e2e8f0", 1.2) } }
            OnSignal("close1") { Actions { setScale(2) } }
            OnSignal("close2") { Actions { setScale(3) } }
            OnSignal("calm") { Actions { setScale(1) setColor("4464748b") } }
            OnCollision("gull") { Actions { sound("hurt") updateBar("lives", -1) flash("gull", "ff0000", 0.3)
                shake(6, 0.3) setPosition("gull", 430, 580, 0.25) } }
        } }
    Entity { id("stormR") startPosition(800, 0) color("9964748b") Type { rectangle(100, 1200) }
        Events {
            OnSignal("warn") { Actions { flash("e2e8f0", 1.2) } }
            OnSignal("close1") { Actions { setScale(2) } }
            OnSignal("close2") { Actions { setScale(3) } }
            OnSignal("calm") { Actions { setScale(1) setColor("4464748b") } }
            OnCollision("gull") { Actions { sound("hurt") updateBar("lives", -1) flash("gull", "ff0000", 0.3)
                shake(6, 0.3) setPosition("gull", 430, 580, 0.25) } }
        } }
    Entity { id("stormB") startPosition(0, 0) color("9964748b") Type { rectangle(900, 100) }
        Events {
            OnSignal("warn") { Actions { flash("e2e8f0", 1.2) } }
            OnSignal("close1") { Actions { setScale(2) } }
            OnSignal("close2") { Actions { setScale(3) } }
            OnSignal("calm") { Actions { setScale(1) setColor("4464748b") } }
            OnCollision("gull") { Actions { sound("hurt") updateBar("lives", -1) flash("gull", "ff0000", 0.3)
                shake(6, 0.3) setPosition("gull", 430, 580, 0.25) } }
        } }
    Entity { id("stormT") startPosition(0, 1100) color("9964748b") Type { rectangle(900, 100) }
        Events {
            OnSignal("warn") { Actions { flash("e2e8f0", 1.2) } }
            OnSignal("close1") { Actions { setScale(2) } }
            OnSignal("close2") { Actions { setScale(3) } }
            OnSignal("calm") { Actions { setScale(1) setColor("4464748b") } }
            OnCollision("gull") { Actions { sound("hurt") updateBar("lives", -1) flash("gull", "ff0000", 0.3)
                shake(6, 0.3) setPosition("gull", 430, 580, 0.25) } }
        } }

    Entity {
        id("barometer")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            Sequence {
                Phase(1, 7) { Actions { sound("buzz") broadcast("warn") } }
                Phase(1, 1.5) { Actions { sound("thud") shake(8, 0.5) broadcast("close1") updateBar("pressure", 1) } }
                Phase(1, 11) { Actions { sound("buzz") broadcast("warn") } }
                Phase(1, 1.5) { Actions { sound("thud") shake(10, 0.6) broadcast("close2") updateBar("pressure", 1) } }
                Phase(1, 22) { Actions { sound("bell") broadcast("calm") updateBar("pressure", -2) } }
                Phase(1, 5) { Actions {
                    sound("win") burst("gull", 30, "bae6fd") gameOver("-- THE HURRICANE PASSED --") } }
            }
            OnElapsedTime(0.9) {
                repeatCount(-1)
                Actions {
                    randomAction {
                        Actions { spawn("shingle", 0, random(120, 1060)) }
                        Actions { spawn("branch", 830, random(120, 1060)) }
                    }
                    whenBar("pressure", ">=", 1) { Actions { spawn("shingle", 0, random(320, 860)) } }
                    whenBar("pressure", ">=", 2) { Actions { spawn("branch", 830, random(320, 860)) } }
                }
            }
        }
    }

    Entity { id("shingle") template(true) startPosition(0, 0) color("b45309") Type { rectangle(40, 20) }
        Movement { speed(4) AutoMove { direction(1, 0) } }
        Events {
            OnElapsedTime { Actions { updateSpeed(random(3, 5.5)) } }
            OnCollision("gull") { Actions { sound("hurt") updateBar("lives", -1) flash("gull", "ff0000", 0.3)
                burst(10, "b45309") deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        } }
    Entity { id("branch") template(true) startPosition(0, 0) color("65a30d") Type { rectangle(70, 12) }
        Movement { speed(3) AutoMove { direction(-1, 0) } }
        Events {
            OnElapsedTime { Actions { updateSpeed(random(2.5, 4.5)) } }
            OnCollision("gull") { Actions { sound("hurt") updateBar("lives", -1) flash("gull", "ff0000", 0.3)
                burst(10, "65a30d") deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        } }
}
