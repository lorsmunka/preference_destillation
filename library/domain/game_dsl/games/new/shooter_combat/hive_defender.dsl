Window {
    size(8, 12)
    background("fef3c7")
    music("bg_5")
    gravity(0)

    lives(3, "-- STUNG OUT --", "lose")
    Bar("hive") {
        initial(8)
        OnValue(3) { Actions { broadcast("swarm") } }
        OnValue(0) { Actions { sound("lose") shake(8, 0.6) burst("hive", 30, "f59e0b") gameOver("-- THE HIVE FELL --") } }
    }
    Bar("swatted") { initial(0) }

    Wave(20, 1.8) {
        Actions {
            randomAction {
                Actions { spawn("wasp", random(0, 770), 0) }
                Actions { spawn("wasp", 0, random(200, 700)) }
                Actions { spawn("wasp", 770, random(200, 700)) }
            }
        }
    }
    Wave(24, 1.3) {
        Actions {
            randomAction {
                Actions { spawn("wasp", random(0, 770), 0) }
                Actions { spawn("wasp", 0, random(200, 700)) spawn("wasp", 770, random(200, 700)) }
                Actions { spawn("hornet", random(0, 740), 0) }
            }
        }
    }
    Wave(30, 0.9) {
        Actions {
            randomAction {
                Actions { spawn("wasp", random(0, 770), 0) spawn("wasp", random(0, 770), 0) }
                Actions { spawn("wasp", 0, random(200, 700)) }
                Actions { spawn("hornet", 740, random(100, 500)) }
                Actions { spawn("hornet", random(0, 740), 0) }
            }
        }
    }
    Wave(6) { Actions { sound("win") shake(5, 0.5) burst("bee", 30, "fde047") gameOver("-- THE HIVE IS SAFE --") } }

    Entity { id("hive") startPosition(340, 1060) color("f59e0b") Type { polygon(6, 60) }
        Events {
            OnCollision("wasp") { Actions {
                deleteCollider() updateBar("hive", -1) sound("buzz") flash("ef4444", 0.3) shake(5, 0.3) burst(10, "facc15") } }
            OnSignal("swarm") { Actions {
                sound("bell") flash("ffffff", 0.4) spawnHere("guard") spawnHere("guard") spawnHere("guard") } }
        } }

    Entity { id("bee") startPosition(380, 500) color("fde047") Type { circle(22) }
        Movement { speed(0) Control { movementControl(0) } }
        Events {
            OnElapsedTime(0.8) { repeatCount(-1) Actions { spawnHere("stinger") } }
        } }

    joystick("bee", 80, 5)

    Entity { id("stinger") template(true) color("1f2937") Type { triangle(16, "up") }
        Movement { speed(5) chase("wasp") }
        Events { OnElapsedTime(2.2) { Actions { deleteEntity() } } } }

    Entity { id("wasp") template(true) color("eab308") Type { triangle(30, "down") }
        Movement { speed(1.2) chase("hive") }
        Events {
            OnCollision("stinger") { Actions {
                deleteCollider() sound("pop") burst(8, "eab308") updateBar("swatted", 1) deleteEntity() } }
        } }

    Entity { id("hornet") template(true) color("b91c1c") Type { polygon(5, 30) }
        var("hp", 4)
        Movement { speed(1) chase("bee") }
        Events {
            OnCollision("stinger") { Actions {
                deleteCollider() changeVar("hp", -1) flash("ffffff", 0.12) sound("bump")
                whenVar("hp", "<=", 0) { Actions {
                    sound("pop") burst(16, "b91c1c") updateBar("swatted", 1) spawnHere("honey") deleteEntity() } } } }
            OnCollision("bee") { Actions {
                sound("hurt") updateBar("lives", -1) flash("bee", "ff0000", 0.3) shake(6, 0.3) burst(12, "b91c1c") deleteEntity() } }
        } }

    Entity { id("honey") template(true) color("fbbf24") Type { circle(14) }
        Events {
            OnCollision("bee") { Actions { sound("pickup") updateBar("hive", 1) burst(10, "fbbf24") deleteEntity() } }
            OnElapsedTime(6) { Actions { setOpacity(0.4) } }
            OnElapsedTime(8) { Actions { deleteEntity() } }
        } }

    Entity { id("guard") template(true) color("fef08a") Type { circle(16) }
        var("hp", 3)
        Movement { speed(2.5) chase("wasp") }
        Events {
            OnCollision("wasp") { Actions {
                deleteCollider() changeVar("hp", -1) updateBar("swatted", 1) sound("pop") burst(8, "eab308")
                whenVar("hp", "<=", 0) { Actions { burst(10, "fef08a") deleteEntity() } } } }
            OnElapsedTime(12) { Actions { deleteEntity() } }
        } }
}
