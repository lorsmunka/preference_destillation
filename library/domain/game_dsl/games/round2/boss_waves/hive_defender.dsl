Window {
    size(10, 9)
    background("422006")
    music("bg_nature")
    lives(3, "-- THE HIVE IS LOST --", "lose")
    winAt("honey", 25, "-- HIVE SAVED --", "win")

    Bar("honey") { initial(0) }
    Bar("queen") {
        initial(6)
        OnValue(3) { Actions { broadcast("queenRage") } }
    }

    HostilePatrol("scoutBee", "player", "-- STUNG --", 3, "facc15") {
        waypoint(80, 780)
        waypoint(900, 780)
        waypoint(500, 450)
    }
    HostilePatrol("guardBee", "player", "-- STUNG --", 2, "f59e0b") {
        waypoint(900, 120)
        waypoint(100, 120)
        dwell(0.8)
    }

    Wave(10, 1.5) { Actions { spawn("nectar", random(80, 900), random(150, 750)) } }
    Wave(10, 1.0) { Actions { spawn("nectar", random(80, 900), random(150, 750)) } }

    Entity {
        id("player")
        startPosition(470, 400)
        color("f8fafc")
        Type { circle(20) }
        Movement { speed(6) Control { movementControl(0) } }
        Events {
            OnElapsedTime(0.45) { repeatCount(-1) Actions { spawnHere("stinger") } }
        }
    }

    Entity {
        id("queen")
        startPosition(430, 740)
        color("eab308")
        Type { star(6, 55, 30) }
        Events {
            OnCollision("stinger") {
                Actions { updateBar("queen", -1) flash("ffffff", 0.12) burst(6, "eab308") }
            }
            OnSignal("queenRage") {
                Actions { setColor("dc2626") shake(7, 0.5) sound("buzz") }
            }
            Sequence {
                Phase(4, 3) { Actions { spawnHere("sting") } }
                Phase(-1, 1.6) { Actions { spawnHere("sting") spawn("sting", 200, 700) spawn("sting", 800, 700) } }
            }
        }
    }

    Entity {
        id("nectar")
        template(true)
        startPosition(0, 0)
        color("fb923c")
        Type { circle(14) }
        Events {
            OnCollision("player") {
                Actions { updateBar("honey", 1) sound("coin") burst(5, "fb923c") deleteEntity() }
            }
            OnElapsedTime(6) { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("stinger")
        template(true)
        startPosition(0, 0)
        color("e2e8f0")
        Type { rectangle(6, 20) }
        Movement { speed(8) AutoMove { direction(0, 1) } }
        Events {
            OnCollision("queen") { Actions { deleteEntity() } }
            OnCollision("scoutBee") { Actions { deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("sting")
        template(true)
        startPosition(0, 0)
        color("dc2626")
        Type { triangle(22, "down") }
        Movement { speed(2.8) chase("player") }
        Events {
            OnCollision("player") {
                Actions { updateBar("lives", -1) sound("hurt") flash("player", "ff0000", 0.3) burst(8, "dc2626") deleteEntity() }
            }
            OnElapsedTime(5) { Actions { burst(4, "dc2626") deleteEntity() } }
        }
    }
}
