# Swipe to kick the ball past the keeper. Three goals win, three misses lose.
Window {
    size(6, 10)
    background("166534")
    music("bg_2")

    Bar("score") { initial(0) }
    Bar("misses") {
        initial(0)
        OnValue(3) { Actions { sound("lose") gameOver("TOO MANY MISSES") } }
    }
    winAt("score", 3, "YOU SCORED 3!", "win")

    Entity {
        id("net")
        startPosition(150, 880)
        color("e2e8f0")
        Type { rectangle(300, 60) }
        Events {
            OnCollision("ball") {
                Actions {
                    sound("coin")
                    updateBar("score", 1)
                    burst(14, "ffffff")
                    broadcast("reset")
                }
            }
        }
    }
    Entity { id("postLeft") startPosition(130, 820) color("f8fafc") solid(true)
        Type { rectangle(20, 140) } }
    Entity { id("postRight") startPosition(450, 820) color("f8fafc") solid(true)
        Type { rectangle(20, 140) } }
    Entity { id("bar") startPosition(130, 960) color("f8fafc") solid(true)
        Type { rectangle(340, 20) } }

    Entity {
        id("keeper")
        startPosition(200, 830)
        color("facc15")
        Type { rectangle(70, 50) }
        Movement {
            speed(3)
            Path {
                waypoint(170, 830)
                waypoint(360, 830)
            }
        }
    }

    Entity {
        id("ball")
        startPosition(280, 120)
        color("ffffff")
        Type { circle(18) }
        Movement { speed(0) AutoMove { direction(0, 0) } }
        Events {
            OnCollision("keeper") {
                Actions {
                    sound("thud")
                    updateBar("misses", 1)
                    broadcast("reset")
                }
            }
            OnCollision("postLeft") { Actions { sound("clang") bounceBack() } }
            OnCollision("postRight") { Actions { sound("clang") bounceBack() } }
            OnCollision("bar") { Actions { sound("clang") bounceBack() } }
            OnCollision {
                Actions {
                    sound("buzz")
                    updateBar("misses", 1)
                    broadcast("reset")
                }
            }
            OnSignal("reset") {
                Actions {
                    setVelocity(0, 0)
                    setPosition("ball", 280, 120, 0)
                }
            }
        }
    }

    Entity {
        id("kicker")
        startPosition(0, 0)
        color("00000000")
        Type { circle(1) }
        Events {
            OnSwipe("up") { Actions { sound("pop") setVelocity("ball", 0, 750) } }
            OnSwipe("left") { Actions { sound("pop") setVelocity("ball", -170, 720) } }
            OnSwipe("right") { Actions { sound("pop") setVelocity("ball", 170, 720) } }
        }
    }
}
