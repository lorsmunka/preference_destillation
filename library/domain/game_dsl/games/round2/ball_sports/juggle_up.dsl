Window {
    size(6, 10)
    background("312e81")
    gravity(700)
    music("bg_1")

    lives(3, "YOU DROPPED IT", "lose")
    Bar("touches") { initial(0) }
    winAt("touches", 20, "20 TOUCHES! NICE", "win")

    Entity {
        id("floor")
        startPosition(0, 0)
        color("1e1b4b")
        Type { rectangle(600, 30) }
        Events {
            OnCollision("ball") {
                Actions {
                    sound("hurt")
                    updateBar("lives", -1)
                    setPosition("ball", 270, 600, 0)
                    setVelocity("ball", 0, 0)
                }
            }
        }
    }

    Entity {
        id("ball")
        startPosition(270, 600)
        color("fb923c")
        Type { circle(34) }
        Movement { speed(0) AutoMove { direction(0, 0) } }
        Events {
            OnTapped {
                Actions {
                    sound("bump")
                    updateBar("touches", 1)
                    setVelocity(0, 640)
                    burst(6, "fde68a")
                }
            }
            OnCollision { Actions { sound("blip") bounceBack() } }
        }
    }

    Entity {
        id("wind")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSwipe("left") { Actions { applyImpulse("ball", -220, 0) } }
            OnSwipe("right") { Actions { applyImpulse("ball", 220, 0) } }
            OnSwipe("up") { Actions { applyImpulse("ball", 0, 160) } }
        }
    }
}
