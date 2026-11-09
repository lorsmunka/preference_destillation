Window {
    size(8, 12)
    background("020617")
    music("bg_4")
    lives(3, "THE ROCKET BROKE UP", "lose")
    Entity {
        id("rocket")
        startPosition(360, 60)
        color("f8fafc")
        Type { triangle(50, "up") }
        Movement { speed(6) Control { movementControl(2) } }
    }
    Entity {
        id("rock")
        template(true)
        startPosition(0, 0)
        color("78716c")
        Type { circle(30) }
        Movement { speed(3) AutoMove { direction(0, -1) } }
        Events {
            OnCollision("rocket") {
                Actions {
                    sound("hurt")
                    shake(6, 0.3)
                    updateBar("lives", -1)
                    deleteEntity()
                }
            }
            OnCollision { Actions { deleteEntity() } }
        }
    }
    Entity {
        id("sky")
        startPosition(0, 1150)
        color("00000000")
        Type { circle(1) }
        Events {
            OnElapsedTime(0.9) {
                repeatCount(-1)
                Actions { spawn("rock", random(40, 720), 1130) }
            }
            OnElapsedTime(25) {
                Actions {
                    whenBar("lives", ">", 0) { Actions { sound("win") gameOver("YOU REACHED THE MOON") } }
                }
            }
        }
    }
    Entity {
        id("label")
        startPosition(20, 1140)
        color("94a3b8")
        Type { text("Survive 25 seconds", 28) }
    }
}
