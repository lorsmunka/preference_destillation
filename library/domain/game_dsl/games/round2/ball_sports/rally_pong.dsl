Window {
    size(7, 11)
    background("0b1020")
    music("bg_4")

    lives(3, "THE RIVAL WINS", "lose")
    Bar("score") { initial(0) }
    winAt("score", 5, "YOU BEAT THE RIVAL", "win")

    paddle("paddle", 280, 70, 140, 24)

    Entity {
        id("rival")
        startPosition(100, 1000)
        color("f43f5e")
        Type { rectangle(120, 24) }
        Movement {
            speed(3.5)
            Path {
                waypoint(40, 1000)
                waypoint(540, 1000)
            }
        }
    }

    Entity {
        id("goalTop")
        startPosition(0, 1090)
        color("00000000")
        Type { rectangle(700, 10) }
    }
    Entity {
        id("goalBottom")
        startPosition(0, 0)
        color("00000000")
        Type { rectangle(700, 10) }
    }

    Entity {
        id("ball")
        startPosition(340, 500)
        color("fde047")
        Type { circle(16) }
        Movement { speed(4) AutoMove { direction(1, -1) } }
        Events {
            OnCollision("paddle") { Actions { sound("blip") bounceBack() } }
            OnCollision("rival") { Actions { sound("blip") bounceBack(0) } }
            OnCollision("goalTop") {
                Actions {
                    sound("coin")
                    updateBar("score", 1)
                    setPosition("ball", 340, 500, 0)
                    setVelocity(150, -400)
                }
            }
            OnCollision("goalBottom") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    setPosition("ball", 340, 500, 0)
                    setVelocity(-150, -400)
                }
            }
            OnCollision { Actions { bounceBack() } }
        }
    }
}
