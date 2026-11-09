# Tap FIRE to shoot the ships coming across the bay. A flagship arrives with each wave.
Window {
    size(12, 7)
    background("0c4a6e")
    music("bg_nature")
    lives(4, "-- HARBOR LOST --", "lose")
    winAt("sunk", 20, "-- BAY SECURED --", "win")

    Bar("sunk") { initial(0) }
    Bar("flagship") {
        initial(5)
        OnValue(0) { Actions { broadcast("flagshipDown") } }
    }

    Wave(10, 2) { Actions { spawn("skiff", 1180, random(80, 600)) } }
    Wave(10, 1.3) { Actions { spawn("skiff", 1180, random(80, 600)) } }
    Wave(6, 6) { Actions { spawn("flagshipBoat", 1150, 330) } }

    Entity { id("dock") startPosition(0, 0) color("78350f") Type { rectangle(60, 700) }
        Events {
            OnCollision("skiff") { Actions { updateBar("lives", -1) sound("thud") shake(5, 0.25) } }
            OnCollision("flagshipBoat") { Actions { updateBar("lives", -2) sound("thud") shake(8, 0.4) } }
        }
    }

    Entity {
        id("cannon")
        startPosition(80, 330)
        color("e5e7eb")
        Type { rectangle(60, 40) }
        Movement { speed(5) Control { movementControl(1) } }
        Events {
            OnSignal("fire") {
                Actions {
                    sound("laser")
                    spawnHere("shell")
                }
            }
        }
    }

    Entity {
        id("fireButton")
        startPosition(1040, 20)
        color("ef4444")
        Type { circle(50) }
        Events {
            OnTapped {
                Actions {
                    broadcast("fire")
                    flash("fecaca", 0.1)
                }
            }
        }
    }

    Entity {
        id("shell")
        template(true)
        startPosition(0, 0)
        color("fbbf24")
        Type { circle(10) }
        Movement { speed(8) AutoMove { direction(1, 0) } }
        Events {
            OnCollision("skiff") { Actions { deleteEntity() } }
            OnCollision("flagshipBoat") {
                Actions {
                    updateBar("flagship", -1)
                    flash("flagshipBoat", "ffffff", 0.15)
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("skiff")
        template(true)
        startPosition(0, 0)
        color("f1f5f9")
        Type { triangle(44, "left") }
        Movement { speed(1.8) AutoMove { direction(-1, 0) } }
        Events {
            OnCollision("shell") {
                Actions {
                    updateBar("sunk", 1)
                    sound("pop")
                    burst(10, "bae6fd")
                    deleteEntity()
                }
            }
            OnCollision("dock") { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("flagshipBoat")
        template(true)
        startPosition(0, 0)
        color("b91c1c")
        Type { rectangle(150, 80) }
        Movement { speed(0.9) AutoMove { direction(-1, 0) } }
        Events {
            OnElapsedTime(1.5) {
                repeatCount(-1)
                Actions { spawnHere("cannonball") }
            }
            OnSignal("flagshipDown") {
                Actions {
                    updateBar("sunk", 5)
                    burst(24, "fb923c")
                    shake(7, 0.5)
                    sound("win2")
                    deleteEntity()
                }
            }
            OnCollision("dock") { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("cannonball")
        template(true)
        startPosition(0, 0)
        color("1f2937")
        Type { circle(14) }
        Movement { speed(4) AutoMove { direction(-1, 0) } }
        Events {
            OnCollision("cannon") {
                Actions {
                    updateBar("lives", -1)
                    flash("cannon", "ff0000", 0.3)
                    sound("hurt")
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }
}
