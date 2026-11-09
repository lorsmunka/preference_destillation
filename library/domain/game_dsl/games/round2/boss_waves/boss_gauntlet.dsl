Window {
    size(8, 12)
    background("111827")
    music("bg_1")
    lives(4, "-- GAUNTLET FAILED --", "lose")

    Bar("slimeHp") {
        initial(10)
        OnValue(0) { Actions { broadcast("slimeDown") } }
    }
    Bar("golemHp") {
        initial(14)
        OnValue(7) { Actions { broadcast("golemEnraged") } }
        OnValue(0) { Actions { broadcast("golemDown") } }
    }

    Level {
        Entity {
            id("player")
            startPosition(370, 80)
            color("4ade80")
            Type { circle(26) }
            Movement { speed(6.5) Control { movementControl(0) } }
            Events {
                OnElapsedTime(0.35) { repeatCount(-1) Actions { spawnHere("bolt") } }
            }
        }
        Entity {
            id("slime")
            startPosition(300, 950)
            color("84cc16")
            Type { circle(90) }
            Movement { speed(2.5) Path { waypoint(80, 950) waypoint(520, 950) } }
            Events {
                OnCollision("bolt") { Actions { updateBar("slimeHp", -1) flash("ffffff", 0.08) } }
                OnSignal("slimeDown") {
                    Actions {
                        sound("win")
                        burst(24, "84cc16")
                        shake(6, 0.4)
                        delay(0.8) { Actions { nextLevel() } }
                    }
                }
                Sequence {
                    Phase(-1, 1.8) { Actions { spawnHere("glob") } }
                }
            }
        }
        Entity {
            id("bolt")
            template(true)
            startPosition(0, 0)
            color("bbf7d0")
            Type { rectangle(8, 24) }
            Movement { speed(9) AutoMove { direction(0, 1) } }
            Events {
                OnCollision("slime") { Actions { deleteEntity() } }
                OnCollision { Actions { deleteEntity() } }
            }
        }
        Entity {
            id("glob")
            template(true)
            startPosition(0, 0)
            color("a3e635")
            Type { circle(20) }
            Movement { speed(3) AutoMove { direction(0, -1) } }
            Events {
                OnCollision("player") {
                    Actions { updateBar("lives", -1) sound("hurt") flash("player", "ff0000", 0.25) burst(8, "a3e635") deleteEntity() }
                }
                OnCollision { Actions { deleteEntity() } }
            }
        }
    }

    Level {
        Entity {
            id("player")
            startPosition(370, 80)
            color("4ade80")
            Type { circle(26) }
            Movement { speed(6.5) Control { movementControl(0) } }
            Events {
                OnElapsedTime(0.35) { repeatCount(-1) Actions { spawnHere("bolt") } }
            }
        }
        Entity {
            id("golem")
            startPosition(280, 940)
            color("94a3b8")
            Type { rectangle(240, 150) }
            Movement { speed(1.2) Path { waypoint(40, 940) waypoint(520, 940) } }
            Events {
                OnCollision("bolt") { Actions { updateBar("golemHp", -1) flash("ffffff", 0.08) } }
                OnSignal("golemEnraged") {
                    Actions { setColor("ef4444") shake(8, 0.6) sound("buzz") spawn("boulder", 350, 800) }
                }
                OnSignal("golemDown") {
                    Actions { sound("win2") burst(30, "94a3b8") shake(10, 0.8) gameOver("-- GAUNTLET CLEARED --") }
                }
                Sequence {
                    Phase(3, 2.4) { Actions { spawn("boulder", 200, 800) spawn("boulder", 520, 800) } }
                    Phase(-1, 1.5) { Actions { spawn("boulder", 120, 800) spawn("boulder", 360, 800) spawn("boulder", 600, 800) } }
                }
            }
        }
        Entity {
            id("bolt")
            template(true)
            startPosition(0, 0)
            color("bbf7d0")
            Type { rectangle(8, 24) }
            Movement { speed(9) AutoMove { direction(0, 1) } }
            Events {
                OnCollision("golem") { Actions { deleteEntity() } }
                OnCollision { Actions { deleteEntity() } }
            }
        }
        Entity {
            id("boulder")
            template(true)
            startPosition(0, 0)
            color("78716c")
            Type { polygon(6, 30) }
            Movement { speed(2.6) AutoMove { direction(0, -1) } }
            Events {
                OnCollision("player") {
                    Actions { updateBar("lives", -1) sound("thud") shake(6, 0.3) burst(10, "78716c") deleteEntity() }
                }
                OnCollision { Actions { deleteEntity() } }
            }
        }
    }
}
