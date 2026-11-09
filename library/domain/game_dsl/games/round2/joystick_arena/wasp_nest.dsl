# A bear raids the clearing for honey while the nest sends out wasps in random directions.
Window {
    size(9, 9)
    background("422006")
    music("bg_nature")
    gravity(0)

    lives(3, "-- STUNG TOO MUCH --", "lose")
    winAt("honey", 10, "-- A WHOLE WINTER OF HONEY --", "win")
    joystick("bear", 75, 5)
    wallBox(0, 0, 900, 900, "713f12", 14)

    Entity {
        id("bear")
        startPosition(430, 300)
        color("92400e")
        Type { circle(34) }
        Movement { speed(0) Control { movementControl(0) } }
    }

    Entity {
        id("nest")
        startPosition(400, 770)
        color("a16207")
        Type { polygon(8, 45) }
        Events {
            OnElapsedTime(2.5) {
                repeatCount(-1)
                Actions {
                    sound("buzz")
                    burst(10, "facc15")
                    spawnHere("wasp")
                    delay(0.4) { Actions { spawnHere("wasp") } }
                }
            }
        }
    }

    Entity {
        id("wasp")
        template(true)
        startPosition(0, 0)
        color("facc15")
        Type { circle(12) }
        Movement { speed(1) AutoMove { direction(0, -1) } }
        Events {
            OnElapsedTime(0.05) {
                Actions { setVelocity(random(-320, 320), random(-320, -60)) }
            }
            OnElapsedTime(1.5) {
                repeatCount(5)
                Actions { setVelocity(random(-340, 340), random(-340, 340)) }
            }
            OnElapsedTime(9.0) {
                Actions {
                    burst(6, "facc15")
                    deleteEntity()
                }
            }
            OnCollision("bear") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("bear", "ef4444", 0.25)
                    shake(5, 0.3)
                    deleteEntity()
                }
            }
            OnCollision { Actions { bounceBack(0) } }
        }
    }

    Entity {
        id("pot")
        template(true)
        startPosition(0, 0)
        color("fbbf24")
        Type { rectangle(30, 34) }
        Events {
            OnCollision("bear") {
                Actions {
                    sound("pickup")
                    updateBar("honey", 1)
                    burst(8, "fbbf24")
                    deleteEntity()
                }
            }
            OnElapsedTime(8.0) { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("forest")
        startPosition(450, 40)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.1) {
                Actions {
                    spawn("pot", 150, 500)
                    spawn("pot", 700, 450)
                    spawn("pot", 430, 600)
                }
            }
            OnElapsedTime(2.0) {
                repeatCount(-1)
                Actions { spawn("pot", random(60, 800), random(250, 700)) }
            }
        }
    }
}
