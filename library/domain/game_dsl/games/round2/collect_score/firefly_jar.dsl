Window {
    size(8, 12)
    background("0b1026")
    music("bg_ambient")
    gravity(0)

    Bar("flies") { initial(0) OnValue(30) { Actions { sound("win") gameOver("-- THE JAR GLOWS --") } } }
    Bar("streak") { initial(0) }
    Bar("lives") { initial(3) OnValue(0) { Actions { sound("lose") gameOver("-- THE NIGHT WON --") } } }

    Entity {
        id("jar")
        startPosition(360, 80)
        color("bae6fd")
        Type { ring(50, 36) }
        Movement { speed(6) Control { movementControl(0) } }
        Events {
            OnElapsedTime(1.5) {
                repeatCount(-1)
                Actions {
                    whenBar("streak", ">", 0) { Actions { updateBar("streak", -1) } }
                }
            }
        }
    }

    Entity {
        id("fly")
        template(true)
        startPosition(0, 0)
        color("d9f99d")
        Type { circle(12) }
        Movement { speed(2) AutoMove { direction(1, 1) } }
        Events {
            OnCollision("jar") {
                Actions {
                    sound("pop")
                    updateBar("flies", 1)
                    updateBar("streak", 1)
                    whenBar("streak", ">=", 3) {
                        Actions {
                            loopCount(2)
                            updateBar("flies", 1)
                            updateBar("streak", -3)
                            sound("sparkle")
                            burst(16, "fde047")
                        }
                    }
                    deleteEntity()
                }
            }
            OnCollision {
                Actions {
                    randomAction {
                        Actions { setVelocity(180, 140) }
                        Actions { setVelocity(-160, 200) }
                        Actions { setVelocity(120, -170) }
                        Actions { setVelocity(-200, -120) }
                    }
                }
            }
            OnElapsedTime(9) {
                Actions {
                    sound("blip")
                    updateBar("lives", -1)
                    deleteEntity()
                }
            }
        }
    }

    Entity { id("moth") startPosition(100, 1000) color("a855f7") Type { triangle(40, "down") }
        Movement { speed(2.2) chase("jar") }
        Events { OnCollision("jar") { Actions { sound("hurt") shake(5, 0.3)
            updateBar("lives", -1) setPosition("moth", 100, 1000, 0) } } } }

    Wave(45, 1.5) { Actions { spawn("fly", random(60, 700), random(500, 1100)) } }
}
