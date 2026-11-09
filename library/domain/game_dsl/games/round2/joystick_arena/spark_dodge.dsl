# Landscape arena. Four turrets in the corners fire sparks across the floor. Drag the drone and last 40 seconds.
Window {
    size(12, 8)
    background("0b132b")
    music("bg_5")
    gravity(0)

    lives(3, "-- DRONE FRIED --", "lose")
    Bar("score") { initial(0) }
    wallBox(0, 0, 1200, 800, "1c2541", 14)

    Entity {
        id("drone")
        startPosition(580, 380)
        color("5bc0be")
        Type { diamond(44, 44) }
        Movement { speed(6.5) Control { movementControl(0) } }
        Events {
            OnElapsedTime(40.0) {
                Actions {
                    whenBar("lives", ">", 0) {
                        Actions {
                            sound("win")
                            shake(5, 0.4)
                            burst(24, "5bc0be")
                            gameOver("-- POWER SURVIVED --")
                        }
                    }
                }
            }
        }
    }

    Entity {
        id("sparkDown")
        template(true)
        startPosition(0, 0)
        color("fde047")
        Type { circle(12) }
        Movement { speed(3.6) AutoMove { direction(1, -0.7) } }
        Events {
            OnCollision("drone") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("drone", "ef4444", 0.25)
                    shake(6, 0.3)
                    burst(10, "fde047")
                    deleteEntity()
                }
            }
            OnCollision { Actions { updateBar("score", 1) deleteEntity() } }
        }
    }

    Entity {
        id("sparkBack")
        template(true)
        startPosition(0, 0)
        color("fb923c")
        Type { rectangle(26, 8) }
        Movement { speed(4.2) AutoMove { direction(-1, -0.4) } }
        Events {
            OnCollision("drone") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("drone", "ef4444", 0.25)
                    shake(6, 0.3)
                    burst(10, "fb923c")
                    deleteEntity()
                }
            }
            OnCollision { Actions { updateBar("score", 1) deleteEntity() } }
        }
    }

    Entity {
        id("sparkUp")
        template(true)
        startPosition(0, 0)
        color("c084fc")
        Type { circle(10) }
        Movement { speed(5) AutoMove { direction(0, 1) } }
        Events {
            OnCollision("drone") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("drone", "ef4444", 0.25)
                    shake(6, 0.3)
                    burst(10, "c084fc")
                    deleteEntity()
                }
            }
            OnCollision { Actions { updateBar("score", 1) deleteEntity() } }
        }
    }

    Entity {
        id("turretLeft")
        startPosition(30, 700)
        color("ef4444")
        Type { rectangle(50, 50) }
        Events {
            OnElapsedTime(1.6) {
                repeatCount(-1)
                Actions { sound("laser") spawnHere("sparkDown") }
            }
        }
    }

    Entity {
        id("turretRight")
        startPosition(1120, 700)
        color("f97316")
        Type { rectangle(50, 50) }
        Events {
            OnElapsedTime(2.1) {
                repeatCount(-1)
                Actions { sound("laser") spawnHere("sparkBack") }
            }
        }
    }

    Entity {
        id("turretFloor")
        startPosition(560, 30)
        color("a855f7")
        Type { triangle(50, "up") }
        Events {
            OnElapsedTime(1.2) {
                repeatCount(-1)
                Actions { spawn("sparkUp", random(100, 1100), 40) }
            }
        }
    }
}
