# Lead the flock into the pen. Sheep trail behind the dog; wolves go for the dog and the stragglers.
Window {
    size(9, 12)
    background("365314")
    music("bg_nature")
    gravity(0)

    lives(3, "-- THE DOG GAVE UP --", "lose")
    winAt("penned", 5, "-- FLOCK SAFE --", "win")
    Bar("lost") {
        initial(0)
        OnValue(3) { Actions { gameOver("-- TOO MANY SHEEP LOST --") } }
    }
    joystick("dog", 80, 5.5)

    Entity {
        id("dog")
        startPosition(200, 400)
        color("92400e")
        Type { circle(24) }
        Movement { speed(0) Control { movementControl(0) } }
    }

    Entity {
        id("pen")
        startPosition(700, 960)
        color("a16207")
        Type { rectangle(180, 180) }
        Events {
            OnCollision("sheep") {
                Actions {
                    sound("bell")
                    updateBar("penned", 1)
                    burst(8, "fef3c7")
                    deleteCollider()
                }
            }
        }
    }
    Entity { id("fenceTop") startPosition(700, 1140) color("78350f") solid(true) Type { rectangle(180, 16) } }
    Entity { id("fenceLeft") startPosition(684, 960) color("78350f") solid(true) Type { rectangle(16, 180) } }

    Entity {
        id("sheep")
        template(true)
        startPosition(0, 0)
        color("fef3c7")
        Type { circle(20) }
        Movement { speed(3) chase("dog", 1.1) }
    }

    Entity {
        id("wolf")
        template(true)
        startPosition(0, 0)
        color("475569")
        Type { triangle(40, "up") }
        Movement { speed(2.3) chase("dog") }
        Events {
            OnCollision("dog") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("dog", "ef4444", 0.25)
                    shake(5, 0.3)
                    burst(10, "475569")
                    deleteEntity()
                }
            }
            OnCollision("sheep") {
                Actions {
                    sound("thud")
                    updateBar("lost", 1)
                    burst(8, "fef3c7")
                    deleteCollider()
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("farm")
        startPosition(450, 40)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.1) {
                Actions {
                    spawn("sheep", 120, 700)
                    spawn("sheep", 300, 820)
                    spawn("sheep", 480, 640)
                    spawn("sheep", 640, 800)
                    spawn("sheep", 400, 900)
                    spawn("sheep", 220, 1000)
                }
            }
            OnElapsedTime(6.0) {
                repeatCount(-1)
                Actions {
                    sound("laser")
                    randomAction {
                        Actions { spawn("wolf", 20, 1150) }
                        Actions { spawn("wolf", 860, 100) }
                        Actions { spawn("wolf", 860, 600) }
                    }
                }
            }
        }
    }
}
