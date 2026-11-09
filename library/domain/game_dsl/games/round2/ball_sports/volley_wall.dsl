Window {
    size(8, 12)
    background("422006")
    gravity(520)
    music("bg_6")

    lives(3, "THE BALL TOUCHED THE SAND", "lose")
    Bar("volleys") { initial(0) }
    winAt("volleys", 15, "15 VOLLEYS IN A ROW", "win")

    paddle("racket", 300, 90, 200, 24)

    Entity {
        id("sand")
        startPosition(0, 0)
        color("a16207")
        Type { rectangle(800, 40) }
        Events {
            OnCollision("ball") {
                Actions {
                    sound("thud")
                    updateBar("lives", -1)
                    setPosition("ball", 400, 800, 0)
                    setVelocity("ball", 100, 0)
                }
            }
        }
    }
    Entity {
        id("net")
        startPosition(380, 40)
        color("f1f5f9")
        solid(true)
        Type { rectangle(16, 260) }
    }

    Entity {
        id("ball")
        startPosition(400, 800)
        color("fef08a")
        Type { circle(26) }
        Movement { speed(0) AutoMove { direction(0, 0) } }
        Events {
            OnCollision("racket") {
                Actions {
                    sound("bump")
                    updateBar("volleys", 1)
                    setVelocity(0, 620)
                    applyImpulse(random(-180, 180), 0)
                }
            }
            OnCollision("net") { Actions { sound("clang") bounceBack() } }
            OnCollision { Actions { bounceBack() } }
        }
    }
}
