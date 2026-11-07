Window {
    size(10, 7)
    background("1e3a2f")
    music("bg_nature")
    gravity(1000)

    lives(3, "-- BONKED --", "lose")
    Bar("clock") { initial(0) }
    winAt("clock", 60, "-- SAFE IN THE NEST --", "win")

    Entity {
        id("canopy")
        startPosition(250, 620)
        color("14532d")
        Type { rectangle(500, 80) }
        Events {
            OnElapsedTime { Actions { ramp("clock", 0, 60, 60) } }
            OnElapsedTime(0.9) {
                repeatCount(-1)
                Actions {
                    whenBar("clock", ">=", 10) { Actions { spawn("cone", random(20, 960), 570) } }
                    whenBar("clock", ">=", 40) { Actions { spawn("cone", random(20, 960), 570) } }
                }
            }
            OnElapsedTime(random(1.4, 2.6)) {
                repeatCount(-1)
                Actions {
                    randomAction {
                        Actions { spawn("logR", 0, 0) }
                        Actions { spawn("logL", 930, 0) }
                    }
                }
            }
        }
    }

    Entity {
        id("squirrel")
        startPosition(478, 0)
        color("d97706")
        Type { rectangle(44, 44) }
        Movement { speed(6) Control { movementControl(2) } }
        var("air", 0)
        Events {
            OnSignal("jump") {
                Actions {
                    whenVar("air", "==", 0) {
                        Actions {
                            changeVar("air", 1)
                            setVelocity(0, 620)
                            sound("jump")
                            burst(6, "fde68a")
                            delay(1.1) { Actions { changeVar("air", -1) } }
                        }
                    }
                }
            }
        }
    }

    Entity {
        id("logR")
        template(true)
        startPosition(0, 0)
        color("92400e")
        gravity(0)
        Type { rectangle(70, 46) }
        Movement { speed(3.2) AutoMove { direction(1, 0) } }
        Events {
            OnElapsedTime { Actions { whenBar("clock", ">=", 30) { Actions { updateSpeed(4.6) } } } }
            OnCollision("squirrel") { Actions {
                sound("thud") updateBar("lives", -1) flash("squirrel", "ff0000", 0.3)
                shake(7, 0.3) burst(14, "a16207") deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("logL")
        template(true)
        startPosition(0, 0)
        color("92400e")
        gravity(0)
        Type { rectangle(70, 46) }
        Movement { speed(3.2) AutoMove { direction(-1, 0) } }
        Events {
            OnElapsedTime { Actions { whenBar("clock", ">=", 30) { Actions { updateSpeed(4.6) } } } }
            OnCollision("squirrel") { Actions {
                sound("thud") updateBar("lives", -1) flash("squirrel", "ff0000", 0.3)
                shake(7, 0.3) burst(14, "a16207") deleteEntity() } }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("cone")
        template(true)
        startPosition(0, 0)
        color("a3e635")
        Type { triangle(30, "down") }
        Movement { speed(1) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("squirrel") { Actions {
                sound("pop") updateBar("lives", -1) flash("squirrel", "ff0000", 0.3)
                burst(10, "a3e635") deleteEntity() } }
            OnCollision { Actions { burst(4, "65a30d") deleteEntity() } }
        }
    }

    Entity { id("padL") startPosition(0, 610) color("374151") Type { rectangle(240, 90) }
        Events { OnTapped("padL") { Actions { broadcast("jump") flash("94a3b8", 0.1) } } } }
    Entity { id("padR") startPosition(760, 610) color("374151") Type { rectangle(240, 90) }
        Events { OnTapped("padR") { Actions { broadcast("jump") flash("94a3b8", 0.1) } } } }
    Entity { id("lblL") startPosition(75, 640) color("e5e7eb") Type { text("JUMP", 30) } }
    Entity { id("lblR") startPosition(835, 640) color("e5e7eb") Type { text("JUMP", 30) } }
}
