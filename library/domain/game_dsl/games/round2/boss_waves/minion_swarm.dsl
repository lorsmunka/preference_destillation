# Tap the minions before they reach the keep. The warlord keeps calling more.
Window {
    size(8, 12)
    background("1c1917")
    music("bg_2")
    lives(5, "-- KEEP OVERRUN --", "lose")
    winAt("score", 30, "-- THE WARLORD RETREATS --", "win")

    Bar("score") { initial(0) }

    Wave(10, 1.4) { Actions { spawn("minion", random(60, 700), 1100) } }
    Wave(10, 1.0) { Actions { spawn("minion", random(60, 700), 1100) } }
    Wave(10, 0.7) {
        Actions {
            spawn("minion", random(60, 700), 1100)
            spawn("brute", random(60, 700), 1100)
        }
    }

    Entity {
        id("keep")
        startPosition(0, 0)
        color("57534e")
        Type { rectangle(800, 90) }
        Events {
            OnCollision("minion") {
                Actions { updateBar("lives", -1) sound("thud") shake(5, 0.2) flash("ef4444", 0.2) }
            }
            OnCollision("brute") {
                Actions { updateBar("lives", -2) sound("thud") shake(8, 0.3) flash("ef4444", 0.3) }
            }
        }
    }

    Entity {
        id("warlord")
        startPosition(310, 1080)
        color("7c3aed")
        Type { polygon(8, 60) }
        Movement { speed(1.5) Path { waypoint(60, 1080) waypoint(620, 1080) } }
        Events {
            Sequence {
                Phase(4, 2.5) { Actions { spawnHere("minion") } }
                Phase(-1, 1.5) {
                    Actions {
                        spawnHere("minion")
                        whenBar("score", ">=", 15) { Actions { spawnHere("brute") } }
                    }
                }
            }
        }
    }

    Entity {
        id("minion")
        template(true)
        startPosition(0, 0)
        color("a3e635")
        Type { circle(32) }
        Movement { speed(1.3) AutoMove { direction(0, -1) } }
        Events {
            OnTapped {
                Actions {
                    updateBar("score", 1)
                    sound("pop")
                    burst(8, "a3e635")
                    deleteEntity()
                }
            }
            OnCollision("keep") { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("brute")
        template(true)
        startPosition(0, 0)
        color("ea580c")
        Type { rectangle(70, 70) }
        var("hp", 3)
        Movement { speed(0.9) AutoMove { direction(0, -1) } }
        Events {
            OnTapped {
                Actions {
                    changeVar("hp", -1)
                    flash("ffffff", 0.1)
                    sound("bump")
                    whenVar("hp", "<=", 0) {
                        Actions {
                            updateBar("score", 3)
                            burst(14, "ea580c")
                            deleteEntity()
                        }
                    }
                }
            }
            OnCollision("keep") { Actions { deleteEntity() } }
        }
    }
}
