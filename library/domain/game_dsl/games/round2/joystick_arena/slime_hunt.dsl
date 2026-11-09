# A knight clears a meadow of slimes. The sword swings by itself; you steer.
Window {
    size(8, 12)
    background("14281d")
    music("bg_nature")
    gravity(0)

    lives(3, "-- THE KNIGHT FELL --", "lose")
    winAt("slimes", 12, "-- THE MEADOW IS CLEAR --", "win")
    joystick("knight", 80, 6)
    wallBox(0, 0, 800, 1200, "3f6212", 16)

    Entity {
        id("knight")
        startPosition(380, 600)
        color("e2e8f0")
        Type { circle(28) }
        Movement { speed(0) Control { movementControl(0) } }
        Events {
            OnElapsedTime(0.9) {
                repeatCount(-1)
                Actions {
                    sound("click")
                    spawnHere("slash")
                }
            }
        }
    }

    Entity {
        id("slash")
        template(true)
        startPosition(0, 0)
        color("80fde68a")
        Type { ring(70, 52) }
        Events {
            OnElapsedTime(0.25) { Actions { deleteEntity() } }
        }
    }

    Entity {
        id("slime")
        template(true)
        startPosition(0, 0)
        color("84cc16")
        Type { circle(26) }
        var("hp", 2)
        Movement { speed(1.3) chase("knight") }
        Events {
            OnCollision("slash") {
                Actions {
                    sound("pop")
                    changeVar("hp", -1)
                    flash("ffffff", 0.15)
                    whenVar("hp", "<=", 0) {
                        Actions {
                            burst(10, "a3e635")
                            updateBar("slimes", 1)
                            deleteEntity()
                        }
                    }
                }
            }
            OnCollision("knight") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    flash("knight", "ef4444", 0.25)
                    shake(5, 0.3)
                    burst(8, "ef4444")
                    deleteEntity()
                }
            }
        }
    }

    Entity {
        id("den")
        startPosition(400, 1150)
        color("00000000")
        Type { circle(1) }
        Events {
            Sequence {
                Phase(5, 2.4) {
                    Actions { spawn("slime", random(60, 720), 1100) }
                }
                Phase(-1, 1.5) {
                    Actions {
                        randomAction {
                            Actions { spawn("slime", random(60, 720), 1100) }
                            Actions { spawn("slime", 40, random(300, 1100)) }
                            Actions { spawn("slime", 720, random(300, 1100)) }
                        }
                    }
                }
            }
        }
    }
}
